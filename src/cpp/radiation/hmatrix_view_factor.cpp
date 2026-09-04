#include "hmatrix_view_factor.h"

#include <embree4/rtcore.h>
#include <embree4/rtcore_ray.h>
#include <omp.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <unordered_map>
#include <atomic>
#include <random>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <Eigen/SVD>
#include <Eigen/QR>

// Platform-specific includes for memory query and mmap
#ifdef __APPLE__
#include <mach/mach.h>
#include <sys/sysctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace radiation {

/**
 * @brief Get available system memory in bytes
 * @return Available memory in bytes, or 0 if unable to determine
 */
static size_t getAvailableMemory() {
#ifdef __APPLE__
    // On macOS, use mach API to get free + inactive memory
    mach_port_t host = mach_host_self();
    vm_statistics64_data_t vm_stats;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;

    if (host_statistics64(host, HOST_VM_INFO64,
                          reinterpret_cast<host_info64_t>(&vm_stats),
                          &count) == KERN_SUCCESS) {
        // Available = free + inactive (can be reclaimed)
        vm_size_t page_size;
        host_page_size(host, &page_size);
        size_t available = (vm_stats.free_count + vm_stats.inactive_count) * page_size;
        return available;
    }
    return 0;
#elif defined(__linux__)
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        // Available = free + buffers + cached
        return info.freeram * info.mem_unit;
    }
    return 0;
#else
    return 0;  // Unknown platform
#endif
}

/**
 * @brief Get total system memory in bytes
 * @return Total memory in bytes, or 0 if unable to determine
 */
static size_t getTotalMemory() {
#ifdef __APPLE__
    int64_t memsize;
    size_t len = sizeof(memsize);
    if (sysctlbyname("hw.memsize", &memsize, &len, nullptr, 0) == 0) {
        return static_cast<size_t>(memsize);
    }
    return 0;
#elif defined(__linux__)
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        return info.totalram * info.mem_unit;
    }
    return 0;
#else
    return 0;
#endif
}

// Thread-safe profiling counters
static std::atomic<long long> g_time_view_factor_us{0};
static std::atomic<long long> g_time_svd_us{0};
static std::atomic<long long> g_time_cluster_cull_us{0};
static std::atomic<size_t> g_view_factor_calls{0};
static std::atomic<size_t> g_svd_calls{0};
static std::atomic<size_t> g_cluster_cull_calls{0};
static std::atomic<size_t> g_cluster_culled{0};

// Memory diagnostics for OOM debugging
static std::atomic<size_t> g_aca_attempts{0};
static std::atomic<size_t> g_aca_successes{0};
static std::atomic<size_t> g_aca_fail_small{0};      // Block too small
static std::atomic<size_t> g_aca_fail_maxrank{0};    // Reached max rank
static std::atomic<size_t> g_aca_fail_norank{0};     // Zero rank (no view factors)
static std::atomic<size_t> g_sparse_fallback_calls{0};
static std::atomic<size_t> g_sparse_fallback_skipped{0};  // Skipped due to MAX_SAFE_BLOCK
static std::atomic<size_t> g_total_index_bytes{0};   // Bytes used for row/col indices
static std::atomic<size_t> g_peak_block_indices{0};  // Peak index memory for a single block
static std::atomic<size_t> g_hierarchical_recursions{0};
static std::atomic<size_t> g_aca_fail_to_zero{0};    // Large admissible blocks treated as zero
static std::atomic<size_t> g_small_blocks_count{0};  // Small blocks (< min_block_size²) stored in RAM
static std::atomic<size_t> g_small_blocks_bytes{0};  // Estimated bytes for small blocks in RAM
// Admissible-block subdivision (robustness fix: bisect instead of zero)
static std::atomic<size_t> g_admissible_subdivided{0};   // Large admissible blocks bisected instead of zeroed
static std::atomic<size_t> g_subdiv_leaves_lowrank{0};   // Sub-blocks captured as ACA low-rank
static std::atomic<size_t> g_subdiv_leaves_sparse{0};    // Sub-blocks computed exact sparse
static std::atomic<size_t> g_subdiv_leaves_zero{0};      // Sub-blocks genuinely empty
static std::atomic<size_t> g_subdiv_depth_cap_zero{0};   // LAST-RESORT: leaf still huge at depth cap -> zeroed

// SVD rank cap: how often tryCompressToLowRank() was asked for a tolerance it
// could not deliver inside max_svd_rank, and returned success anyway.
//
// The truncation rank is chosen from the singular-value spectrum against
// `compression_tolerance`, and then clamped to `max_svd_rank` (50). When the
// clamp binds, the block does NOT meet the requested tolerance -- and the
// function still returns true, so the caller stores the approximation and no
// line of output says the tolerance was abandoned. That silence is the defect;
// the clamp itself is a legitimate memory bound. These counters turn it into a
// number, and PTM_HMATRIX_RANK_CAP_FATAL / radiation.hmatrix_rank_cap_fatal
// turns the number into a stop.
static std::atomic<size_t> g_svd_rank_capped{0};       // blocks where the cap bound
static std::atomic<size_t> g_svd_compress_calls{0};    // blocks that reached the rank choice
// Worst achieved relative residual sigma(rank_cap)/sigma(0) over capped blocks,
// i.e. the tolerance the builder actually delivered where it promised
// compression_tolerance. Stored as a double under compare-exchange.
static std::atomic<double> g_svd_rank_cap_worst_resid{0.0};
// Largest rank the spectrum asked for, over capped blocks.
static std::atomic<int> g_svd_rank_cap_worst_want{0};

// Near pairs the M1 pre-cull WOULD have deleted, and which now reach the
// refined kernel instead. Zero means the change is inert on this mesh -- a
// measured zero, not an assumed one.
static std::atomic<size_t> g_precull_near_bypassed{0};

// Near-field EXPOSURE of the ACA cross generators.
//
// computeViewFactorRow/Column are the matrix-entry oracles ACA samples to build
// a low-rank factorization. They see the same pairs computeViewFactorBlock sees,
// so if a pair is close enough that the M1 centroid estimate is wrong, it is
// wrong here too -- and until 2026-08-23 these two generators were the only VF
// code in the engine with NO near-field branch at all. Whether a close pair got
// the M2 quadrature therefore depended on which BLOCK the octree happened to put
// it in, which is not a physical property of the pair.
//
// These counters exist so the exposure is a measured number rather than an
// argument. A gate that asserts "the cross generators agree with the dense
// generator on near pairs" is vacuous if the cross generators never saw one --
// this is what makes that arm non-vacuous.
// The MARGIN matters as much as the count. On the two meshes measured
// 2026-08-23 the count is zero -- but "zero" alone does not say whether the
// admissibility condition misses the near-field window by a factor of 20 or by
// 5 percent, and the answer decides whether the gap is unreachable or merely
// unreached. So the closest approach is recorded too, as min r / (w d) with the
// CONFIGURED window w -- a fixed 4 here would under-report the moment anyone
// changed nearfield_window_factor.
static std::atomic<size_t> g_cross_pairs{0};        // pairs surviving culling in row/col
static std::atomic<size_t> g_cross_near_pairs{0};   // ... of those, inside the near window
static std::atomic<double> g_cross_min_r_over_4d{1.0e300};

static inline void noteCrossPair(double r, double d4) {
    ++g_cross_pairs;
    const double ratio = (d4 > 0.0) ? r / d4 : 1.0e300;
    if (ratio < 1.0) ++g_cross_near_pairs;
    double prev = g_cross_min_r_over_4d.load(std::memory_order_relaxed);
    while (ratio < prev &&
           !g_cross_min_r_over_4d.compare_exchange_weak(prev, ratio,
                                                        std::memory_order_relaxed)) {
    }
}

// Shared source of truth for the ACA face-count cap (env PTM_HMATRIX_ACA_CAP,
// default 12000 faces/side; 0 = off). Read once. Used both by buildBlock's
// admissible pre-zero guard and by buildAdmissibleSubdivided's per-sub-block gate.
static size_t getAcaCap() {
    static const size_t v = [](){
        const char* e = std::getenv("PTM_HMATRIX_ACA_CAP");
        return e ? std::strtoul(e, nullptr, 10) : 12000UL;
    }();
    return v;
}

// Visibility-ray lift: fraction of the local face edge to lift the occlusion ray
// origin ALONG THE FACE NORMAL, clearing local sub-mesh roughness so grazing
// floor->rim sightlines are not spuriously self-occluded (env PTM_HMATRIX_RAY_LIFT;
// 0 = legacy along-ray offset only). ~0.3-0.5 clears ~10deg facet roughness.
static double getRayLiftFactor() {
    static const double v = [](){
        const char* e = std::getenv("PTM_HMATRIX_RAY_LIFT");
        return e ? std::atof(e) : 0.0;
    }();
    return v;
}

// Whether to bisect+re-ACA large admissible blocks instead of zeroing them.
// Env PTM_HMATRIX_SUBDIVIDE (1/0) overrides the config flag when set; OFF
// reproduces the exact legacy zero behavior for A/B verification.
static bool subdivideEnabled(const HMatrixConfig& c) {
    static const int env = [](){
        const char* e = std::getenv("PTM_HMATRIX_SUBDIVIDE");
        return e ? std::atoi(e) : -1;   // -1 = unset, defer to config
    }();
    return env < 0 ? c.subdivide_large_admissible : (env != 0);
}

static void resetProfilingCounters() {
    g_time_view_factor_us = 0;
    g_time_svd_us = 0;
    g_time_cluster_cull_us = 0;
    g_view_factor_calls = 0;
    g_svd_calls = 0;
    g_cluster_cull_calls = 0;
    g_cluster_culled = 0;
    // Memory diagnostics
    g_aca_attempts = 0;
    g_aca_successes = 0;
    g_aca_fail_small = 0;
    g_aca_fail_maxrank = 0;
    g_aca_fail_norank = 0;
    g_sparse_fallback_calls = 0;
    g_sparse_fallback_skipped = 0;
    g_total_index_bytes = 0;
    g_peak_block_indices = 0;
    g_hierarchical_recursions = 0;
    g_aca_fail_to_zero = 0;
    g_small_blocks_count = 0;
    g_small_blocks_bytes = 0;
    g_admissible_subdivided = 0;
    g_subdiv_leaves_lowrank = 0;
    g_subdiv_leaves_sparse = 0;
    g_subdiv_leaves_zero = 0;
    g_subdiv_depth_cap_zero = 0;
    g_cross_pairs = 0;
    g_cross_near_pairs = 0;
    g_cross_min_r_over_4d = 1.0e300;
    g_svd_rank_capped = 0;
    g_svd_compress_calls = 0;
    g_svd_rank_cap_worst_resid = 0.0;
    g_svd_rank_cap_worst_want = 0;
    g_precull_near_bypassed = 0;
}

static void printProfilingCounters(double total_ms) {
    double vf_ms = g_time_view_factor_us.load() / 1000.0;
    double svd_ms = g_time_svd_us.load() / 1000.0;
    double cull_ms = g_time_cluster_cull_us.load() / 1000.0;

    std::cout << "  Detailed timing (inside block build):" << std::endl;
    std::cout << "    View factor computation: " << std::fixed << std::setprecision(2)
              << vf_ms << " ms (" << (100.0 * vf_ms / total_ms) << "%) - "
              << g_view_factor_calls.load() << " calls" << std::endl;
    std::cout << "    SVD/ACA compression:     " << svd_ms << " ms ("
              << (100.0 * svd_ms / total_ms) << "%) - "
              << g_svd_calls.load() << " calls" << std::endl;
    std::cout << "    Cluster culling:         " << cull_ms << " ms ("
              << (100.0 * cull_ms / total_ms) << "%) - "
              << g_cluster_cull_calls.load() << " calls, "
              << g_cluster_culled.load() << " culled" << std::endl;
    double overhead_ms = total_ms - vf_ms - svd_ms - cull_ms;
    std::cout << "    Other overhead:          " << overhead_ms << " ms ("
              << (100.0 * overhead_ms / total_ms) << "%)" << std::endl;

    // Memory diagnostics
    std::cout << "  Memory diagnostics:" << std::endl;
    std::cout << "    ACA attempts:            " << g_aca_attempts.load() << std::endl;
    std::cout << "    ACA successes:           " << g_aca_successes.load()
              << " (" << std::fixed << std::setprecision(1)
              << (g_aca_attempts > 0 ? 100.0 * g_aca_successes.load() / g_aca_attempts.load() : 0.0)
              << "%)" << std::endl;
    std::cout << "    ACA failures:" << std::endl;
    std::cout << "      - Block too small:     " << g_aca_fail_small.load() << std::endl;
    std::cout << "      - Max rank reached:    " << g_aca_fail_maxrank.load() << std::endl;
    std::cout << "      - Zero rank (no VFs):  " << g_aca_fail_norank.load() << std::endl;
    std::cout << "    Sparse fallback calls:   " << g_sparse_fallback_calls.load() << std::endl;
    std::cout << "    Sparse fallback skipped: " << g_sparse_fallback_skipped.load()
              << " (block too large)" << std::endl;
    std::cout << "    ACA fail → zero:         " << g_aca_fail_to_zero.load()
              << " (large admissible blocks)" << std::endl;
    std::cout << "    Admissible subdivided:   " << g_admissible_subdivided.load()
              << " (leaves: lowrank=" << g_subdiv_leaves_lowrank.load()
              << ", sparse=" << g_subdiv_leaves_sparse.load()
              << ", zero=" << g_subdiv_leaves_zero.load()
              << ", depth-cap-zero=" << g_subdiv_depth_cap_zero.load() << ")" << std::endl;
    std::cout << "    Hierarchical recursions: " << g_hierarchical_recursions.load() << std::endl;
    std::cout << "    ACA cross-generator pairs: " << g_cross_pairs.load()
              << " (near field, r < 4d: " << g_cross_near_pairs.load()
              << "; closest approach r/4d = " << g_cross_min_r_over_4d.load() << ")"
              << std::endl;
    std::cout << "    Small blocks (in RAM):   " << g_small_blocks_count.load()
              << " (" << (g_small_blocks_bytes.load() / (1024.0 * 1024.0)) << " MB)" << std::endl;
    std::cout << "    Index memory usage:      " << std::fixed << std::setprecision(2)
              << (g_total_index_bytes.load() / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "    Peak block indices:      " << std::fixed << std::setprecision(2)
              << (g_peak_block_indices.load() / (1024.0 * 1024.0)) << " MB" << std::endl;
}

// ============================================================================
// QuadtreeNode Implementation
// ============================================================================

void QuadtreeNode::build(const std::vector<std::array<double, 3>>& face_centers,
                         const std::vector<int>& indices,
                         int max_depth, int min_size, int current_depth) {
    depth = current_depth;
    face_indices = indices;

    if (indices.empty()) {
        is_leaf = true;
        return;
    }

    // Compute 2D bounding box (x,y projection, ignoring z for quadtree)
    min_bound = {{face_centers[indices[0]][0], face_centers[indices[0]][1]}};
    max_bound = min_bound;

    for (int idx : indices) {
        const auto& c = face_centers[idx];
        min_bound[0] = std::min(min_bound[0], c[0]);
        min_bound[1] = std::min(min_bound[1], c[1]);
        max_bound[0] = std::max(max_bound[0], c[0]);
        max_bound[1] = std::max(max_bound[1], c[1]);
    }

    center = {{(min_bound[0] + max_bound[0]) / 2.0,
               (min_bound[1] + max_bound[1]) / 2.0}};

    double dx = max_bound[0] - min_bound[0];
    double dy = max_bound[1] - min_bound[1];
    diameter = std::sqrt(dx * dx + dy * dy);

    // Check stopping conditions
    if (current_depth >= max_depth ||
        static_cast<int>(indices.size()) <= min_size) {
        is_leaf = true;
        return;
    }

    // Partition into quadrants
    std::array<std::vector<int>, 4> child_indices;

    for (int idx : indices) {
        const auto& c = face_centers[idx];
        int quadrant = 0;
        if (c[0] >= center[0]) quadrant |= 1;  // East
        if (c[1] >= center[1]) quadrant |= 2;  // North
        child_indices[quadrant].push_back(idx);
    }

    // Check if partitioning is useful (at least 2 non-empty quadrants)
    int non_empty = 0;
    for (int q = 0; q < 4; ++q) {
        if (!child_indices[q].empty()) non_empty++;
    }

    if (non_empty < 2) {
        // Don't subdivide if all points are in one quadrant
        is_leaf = true;
        return;
    }

    // Create children
    is_leaf = false;
    for (int q = 0; q < 4; ++q) {
        if (!child_indices[q].empty()) {
            children[q] = std::make_unique<QuadtreeNode>();
            children[q]->build(face_centers, child_indices[q],
                               max_depth, min_size, current_depth + 1);
        }
    }
}

void QuadtreeNode::getLeaves(std::vector<QuadtreeNode*>& leaves) {
    if (is_leaf) {
        if (!face_indices.empty()) {
            leaves.push_back(this);
        }
        return;
    }

    for (int q = 0; q < 4; ++q) {
        if (children[q]) {
            children[q]->getLeaves(leaves);
        }
    }
}

void QuadtreeNode::getNodesAtDepth(int target_depth, std::vector<QuadtreeNode*>& nodes) {
    if (depth == target_depth || is_leaf) {
        if (!face_indices.empty()) {
            nodes.push_back(this);
        }
        return;
    }

    for (int q = 0; q < 4; ++q) {
        if (children[q]) {
            children[q]->getNodesAtDepth(target_depth, nodes);
        }
    }
}

// ============================================================================
// OctreeNode Implementation (3D spatial partitioning)
// ============================================================================

void OctreeNode::build(const std::vector<std::array<double, 3>>& face_centers,
                       const std::vector<int>& indices,
                       int max_depth, int min_size, int current_depth) {
    depth = current_depth;
    face_indices = indices;

    if (indices.empty()) {
        is_leaf = true;
        return;
    }

    // Compute 3D bounding box
    min_bound = face_centers[indices[0]];
    max_bound = min_bound;

    for (int idx : indices) {
        const auto& c = face_centers[idx];
        min_bound[0] = std::min(min_bound[0], c[0]);
        min_bound[1] = std::min(min_bound[1], c[1]);
        min_bound[2] = std::min(min_bound[2], c[2]);
        max_bound[0] = std::max(max_bound[0], c[0]);
        max_bound[1] = std::max(max_bound[1], c[1]);
        max_bound[2] = std::max(max_bound[2], c[2]);
    }

    center = {{(min_bound[0] + max_bound[0]) / 2.0,
               (min_bound[1] + max_bound[1]) / 2.0,
               (min_bound[2] + max_bound[2]) / 2.0}};

    double dx = max_bound[0] - min_bound[0];
    double dy = max_bound[1] - min_bound[1];
    double dz = max_bound[2] - min_bound[2];
    diameter = std::sqrt(dx * dx + dy * dy + dz * dz);

    // Check stopping conditions
    if (current_depth >= max_depth ||
        static_cast<int>(indices.size()) <= min_size) {
        is_leaf = true;
        return;
    }

    // Partition into 8 octants
    std::array<std::vector<int>, 8> child_indices;

    for (int idx : indices) {
        const auto& c = face_centers[idx];
        int octant = getOctant(c);
        child_indices[octant].push_back(idx);
    }

    // Check if partitioning is useful (at least 2 non-empty octants)
    int non_empty = 0;
    for (int o = 0; o < 8; ++o) {
        if (!child_indices[o].empty()) non_empty++;
    }

    if (non_empty < 2) {
        // Don't subdivide if all points are in one octant
        is_leaf = true;
        return;
    }

    // Create children
    is_leaf = false;
    for (int o = 0; o < 8; ++o) {
        if (!child_indices[o].empty()) {
            children[o] = std::make_unique<OctreeNode>();
            children[o]->build(face_centers, child_indices[o],
                               max_depth, min_size, current_depth + 1);
        }
    }
}

void OctreeNode::getLeaves(std::vector<OctreeNode*>& leaves) {
    if (is_leaf) {
        if (!face_indices.empty()) {
            leaves.push_back(this);
        }
        return;
    }

    for (int o = 0; o < 8; ++o) {
        if (children[o]) {
            children[o]->getLeaves(leaves);
        }
    }
}

void OctreeNode::getNodesAtDepth(int target_depth, std::vector<OctreeNode*>& nodes) {
    if (depth == target_depth || is_leaf) {
        if (!face_indices.empty()) {
            nodes.push_back(this);
        }
        return;
    }

    for (int o = 0; o < 8; ++o) {
        if (children[o]) {
            children[o]->getNodesAtDepth(target_depth, nodes);
        }
    }
}

// ============================================================================
// Block Implementations
// ============================================================================

void SparseBlock::matvec(const double* x, double* y) const {
    for (int i = 0; i < n_rows; ++i) {
        double sum = 0.0;
        for (int k = row_ptrs[i]; k < row_ptrs[i + 1]; ++k) {
            sum += values[k] * x[col_indices[k]];
        }
        y[i] += sum;
    }
}

size_t SparseBlock::memoryBytes() const {
    return values.size() * sizeof(double) +
           col_indices.size() * sizeof(int) +
           row_ptrs.size() * sizeof(int);
}

void LowRankBlock::matvec(const double* x, double* y) const {
    if (rank == 0) return;

    // y += U * diag(sigma) * V^T * x
    // Manual loops avoid allocation overhead for small rank matrices
    // Eigen Map + operations allocate temporaries which hurt performance

    // Step 1: temp = diag(sigma) * V^T * x (rank x 1)
    // Use stack allocation for small ranks (common case)
    constexpr int STACK_THRESHOLD = 64;
    double temp_stack[STACK_THRESHOLD];
    std::vector<double> temp_heap;
    double* temp = (rank <= STACK_THRESHOLD) ? temp_stack :
                   (temp_heap.resize(rank), temp_heap.data());

    for (int r = 0; r < rank; ++r) {
        double sum = 0.0;
        for (int j = 0; j < n_cols; ++j) {
            sum += V(j, r) * x[j];
        }
        temp[r] = sum * sigma(r);
    }

    // Step 2: y += U * temp
    for (int i = 0; i < n_rows; ++i) {
        double sum = 0.0;
        for (int r = 0; r < rank; ++r) {
            sum += U(i, r) * temp[r];
        }
        y[i] += sum;
    }
}

size_t LowRankBlock::memoryBytes() const {
    return U.size() * sizeof(double) +
           sigma.size() * sizeof(double) +
           V.size() * sizeof(double);
}

void DenseBlock::matvec(const double* x, double* y) const {
    // Manual loop - avoids Eigen Map overhead for small blocks
    for (int i = 0; i < n_rows; ++i) {
        double sum = 0.0;
        for (int j = 0; j < n_cols; ++j) {
            sum += data(i, j) * x[j];
        }
        y[i] += sum;
    }
}

size_t DenseBlock::memoryBytes() const {
    return data.size() * sizeof(double);
}

void HierarchicalBlock::matvec(const double* x, double* y) const {
    for (const auto& child : children) {
        if (child) {
            child->matvec(x, y);
        }
    }
}

size_t HierarchicalBlock::memoryBytes() const {
    size_t total = row_indices.size() * sizeof(int) +
                   col_indices.size() * sizeof(int);
    for (const auto& child : children) {
        if (child) {
            total += child->memoryBytes();
        }
    }
    return total;
}

int HMatrixBlock::numRows() const {
    return std::visit([](const auto& block) { return block.n_rows; }, data);
}

int HMatrixBlock::numCols() const {
    return std::visit([](const auto& block) { return block.n_cols; }, data);
}

void HMatrixBlock::matvec(const double* x, double* y) const {
    // Note: x and y are in GLOBAL coordinates
    // Each block type handles the global-to-local mapping differently

    // Special handling for hierarchical blocks - they pass through to children
    // which have their own global index mappings.
    // IMPORTANT: HierarchicalBlocks don't store their own indices (memory optimization),
    // so we check for this FIRST before checking row_indices/col_indices.
    if (std::holds_alternative<HierarchicalBlock>(data)) {
        const auto& hier = std::get<HierarchicalBlock>(data);
        hier.matvec(x, y);  // Pass global arrays directly to children
        return;
    }

    // Special handling for disk-backed blocks - they read data from mmap
    // DiskBlock has a different signature: it needs indices passed in
    if (std::holds_alternative<DiskBlock>(data)) {
        const auto& disk = std::get<DiskBlock>(data);
        disk.matvec(x, y, row_indices, col_indices);  // Works directly on global arrays
        return;
    }

    // For non-hierarchical blocks, we need indices for global-to-local mapping
    if (row_indices.empty() || col_indices.empty()) return;

    // For non-hierarchical blocks: extract local portions, compute, write back
    // Extract x values for our columns
    std::vector<double> x_local(col_indices.size());
    for (size_t j = 0; j < col_indices.size(); ++j) {
        x_local[j] = x[col_indices[j]];
    }

    // Compute local y
    std::vector<double> y_local(row_indices.size(), 0.0);

    std::visit([&x_local, &y_local](const auto& block) {
        block.matvec(x_local.data(), y_local.data());
    }, data);

    // Add local y to global y
    for (size_t i = 0; i < row_indices.size(); ++i) {
        #pragma omp atomic
        y[row_indices[i]] += y_local[i];
    }
}

size_t HMatrixBlock::memoryBytes() const {
    size_t base = row_indices.size() * sizeof(int) +
                  col_indices.size() * sizeof(int);
    return base + std::visit([](const auto& block) {
        return block.memoryBytes();
    }, data);
}

// ============================================================================
// HMatrixViewFactor Implementation
// ============================================================================

// Macroscopic (smoothed) face normals via a LOCAL PLANE FIT (PCA of neighbor face
// centers within `radius`): the eigenvector of the smallest covariance eigenvalue
// is the local surface normal, oriented to agree with the facet normal. This gives
// "up" on the flat floor and the true wall orientation on walls (UNLIKE averaging
// facet normals, which cancels on steep/opposing faces and corrupts the VF). Used
// for the VF cosine terms so grazing floor<->rim coupling survives on fine meshes.
void HMatrixViewFactor::computeSmoothedNormals(double radius) {
    const size_t N = face_centers_.size();
    smoothed_normals_.assign(N, {{0.0, 0.0, 0.0}});
    if (N == 0) return;
    const double inv = 1.0 / std::max(1e-6, radius);
    std::array<double, 3> mn = face_centers_[0], mx = face_centers_[0];
    for (size_t i = 1; i < N; ++i)
        for (int k = 0; k < 3; ++k) {
            mn[k] = std::min(mn[k], face_centers_[i][k]);
            mx[k] = std::max(mx[k], face_centers_[i][k]);
        }
    const long long ny = (long long)((mx[1] - mn[1]) * inv) + 1;
    const long long nz = (long long)((mx[2] - mn[2]) * inv) + 1;
    auto cellkey = [&](const std::array<double, 3>& p) -> long long {
        long long cx = (long long)((p[0] - mn[0]) * inv);
        long long cy = (long long)((p[1] - mn[1]) * inv);
        long long cz = (long long)((p[2] - mn[2]) * inv);
        return (cx * ny + cy) * nz + cz;
    };
    std::unordered_map<long long, std::vector<int>> grid;
    grid.reserve(N);
    for (size_t i = 0; i < N; ++i) grid[cellkey(face_centers_[i])].push_back((int)i);
    const double r2 = radius * radius;
    #pragma omp parallel for schedule(dynamic, 1024)
    for (size_t i = 0; i < N; ++i) {
        const auto& ci = face_centers_[i];
        long long cx = (long long)((ci[0] - mn[0]) * inv);
        long long cy = (long long)((ci[1] - mn[1]) * inv);
        long long cz = (long long)((ci[2] - mn[2]) * inv);
        // accumulate weighted covariance of neighbor centers about their mean
        double sw = 0.0; std::array<double,3> mean = {{0,0,0}};
        std::vector<std::pair<std::array<double,3>,double>> nbr;
        nbr.reserve(64);
        for (int dx = -1; dx <= 1; ++dx)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dz = -1; dz <= 1; ++dz) {
                    long long key = ((cx + dx) * ny + (cy + dy)) * nz + (cz + dz);
                    auto it = grid.find(key);
                    if (it == grid.end()) continue;
                    for (int j : it->second) {
                        const auto& cj = face_centers_[j];
                        double d2 = (cj[0]-ci[0])*(cj[0]-ci[0]) + (cj[1]-ci[1])*(cj[1]-ci[1]) + (cj[2]-ci[2])*(cj[2]-ci[2]);
                        if (d2 > r2) continue;
                        double w = face_areas_[j];
                        nbr.push_back({cj, w}); sw += w;
                        mean[0]+=w*cj[0]; mean[1]+=w*cj[1]; mean[2]+=w*cj[2];
                    }
                }
        if (nbr.size() < 6 || sw <= 0) { smoothed_normals_[i] = face_normals_[i]; continue; }
        mean[0]/=sw; mean[1]/=sw; mean[2]/=sw;
        Eigen::Matrix3d C = Eigen::Matrix3d::Zero();
        for (auto& pr : nbr) {
            double w = pr.second;
            double ex = pr.first[0]-mean[0], ey = pr.first[1]-mean[1], ez = pr.first[2]-mean[2];
            C(0,0)+=w*ex*ex; C(0,1)+=w*ex*ey; C(0,2)+=w*ex*ez;
            C(1,1)+=w*ey*ey; C(1,2)+=w*ey*ez; C(2,2)+=w*ez*ez;
        }
        C(1,0)=C(0,1); C(2,0)=C(0,2); C(2,1)=C(1,2);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(C);
        Eigen::Vector3d nrm = es.eigenvectors().col(0);   // smallest eigenvalue = normal
        std::array<double,3> sn = {{nrm[0], nrm[1], nrm[2]}};
        // orient to agree with the facet normal (outward)
        const auto& fn = face_normals_[i];
        if (sn[0]*fn[0]+sn[1]*fn[1]+sn[2]*fn[2] < 0) { sn[0]=-sn[0]; sn[1]=-sn[1]; sn[2]=-sn[2]; }
        smoothed_normals_[i] = sn;
    }
}

void HMatrixViewFactor::build(const std::shared_ptr<ptm::Mesh>& mesh,
                               RTCDevice device,
                               RTCScene scene,
                               const RadiationConfig& rad_config,
                               const HMatrixConfig& hmat_config) {
    auto start_time = std::chrono::high_resolution_clock::now();
    resetProfilingCounters();

    mesh_ = mesh;
    device_ = device;
    scene_ = scene;
    config_ = hmat_config;
    rad_config_ = rad_config;

    size_t n_faces = mesh_->faces.size();
    n_rows_ = n_faces;
    n_cols_ = n_faces;

    // AUTO-SCALE parameters for very large meshes to prevent block explosion
    // For 2M faces with min_block_size=64, we'd get ~1M+ blocks which is too many
    if (n_faces > 500000) {
        // Scale min_block_size: target ~100-200 leaf blocks per tree dimension
        // With octree depth 3, we have ~339 nodes, each with ~5893 faces
        // min_block_size should be at least sqrt(faces_per_node) to avoid excessive subdivision
        int faces_per_node_estimate = static_cast<int>(n_faces / 400);  // Assume ~400 octree nodes
        int new_min_block_size = std::max(config_.min_block_size,
            static_cast<int>(std::sqrt(static_cast<double>(faces_per_node_estimate))));

        // Cap max_tree_depth to prevent deep recursion
        int new_max_depth = std::min(config_.max_tree_depth, 6);

        // For >1M face meshes, hierarchical subdivision was previously disabled
        // entirely to bound block count. That made the top-of-octree non-admissible
        // blocks (~12k faces/side on Amundsen 4M) undivided, so the sparse-fallback
        // working set ballooned past 300 GB at ~11% build progress before OOM.
        // Clamp to 1 level instead: non-admissible huge blocks get a single 8×8
        // subdivision, capping per-block working memory at ~(n/8)²·56 B while
        // only ~64× multiplying a minority of blocks (admissible blocks are
        // unaffected). Empirically keeps peak RSS bounded on 384 GB machines.
        int new_hierarchical_levels = config_.max_hierarchical_levels;
        if (n_faces > 1000000) {
            new_hierarchical_levels = std::min(new_hierarchical_levels, 1);
        }

        bool any_change = (new_min_block_size > config_.min_block_size ||
                           new_max_depth < config_.max_tree_depth ||
                           new_hierarchical_levels != config_.max_hierarchical_levels);

        if (any_change) {
            std::cout << "  Large mesh optimization (>" << n_faces << " faces):" << std::endl;
            if (new_min_block_size > config_.min_block_size) {
                std::cout << "    Increasing min_block_size: " << config_.min_block_size
                          << " -> " << new_min_block_size << std::endl;
                config_.min_block_size = new_min_block_size;
            }
            if (new_max_depth < config_.max_tree_depth) {
                std::cout << "    Reducing max_tree_depth: " << config_.max_tree_depth
                          << " -> " << new_max_depth << std::endl;
                config_.max_tree_depth = new_max_depth;
            }
            if (new_hierarchical_levels != config_.max_hierarchical_levels) {
                std::cout << "    Disabling hierarchical subdivision: " << config_.max_hierarchical_levels
                          << " -> " << new_hierarchical_levels << " levels" << std::endl;
                config_.max_hierarchical_levels = new_hierarchical_levels;
            }
        }
    }

    // Env override (tuning lever for OOM mitigation): more hierarchical levels
    // subdivide the near-field non-admissible blocks into smaller leaves,
    // shrinking per-block view-factor working memory at the cost of more
    // (mostly cheap) blocks. Overrides the large-mesh clamp above.
    if (const char* env_h = std::getenv("PTM_HMATRIX_HIER_LEVELS")) {
        int h = std::atoi(env_h);
        if (h >= 0) {
            std::cout << "  Env PTM_HMATRIX_HIER_LEVELS=" << h
                      << " (was " << config_.max_hierarchical_levels << ")" << std::endl;
            config_.max_hierarchical_levels = h;
        }
    }

    // Always print basic H-matrix build info (this is important for large meshes)
    std::cout << "Building H-matrix view factor for " << n_faces << " faces..." << std::endl;
    {
        // Show the tolerance in default float notation so an inherited
        // std::fixed / low-precision stream state (from an upstream caller) cannot
        // truncate it to a misleading "0.0". Display-only; the value is honored
        // regardless (BUGS.md PTM-09). Save/restore leaves cout unchanged.
        const std::ios::fmtflags saved_fmt = std::cout.flags();
        const std::streamsize saved_prec = std::cout.precision();
        std::cout.unsetf(std::ios::floatfield);
        std::cout.precision(6);
        std::cout << "  Compression tolerance: " << config_.compression_tolerance << std::endl;
        std::cout.flags(saved_fmt);
        std::cout.precision(saved_prec);
    }
    std::cout << "  Max tree depth: " << config_.max_tree_depth << std::endl;
    std::cout << "  Min block size: " << config_.min_block_size << std::endl;
    std::cout << "  Max SVD rank: " << config_.max_svd_rank << std::endl;
    std::cout << "  Max hierarchical levels: " << config_.max_hierarchical_levels << std::endl;

    // Memory estimation for large meshes
    double estimated_dense_gb = (double(n_faces) * double(n_faces) * sizeof(double)) / (1024.0 * 1024.0 * 1024.0);
    double estimated_sparse_gb = (double(n_faces) * 1000.0 * (sizeof(double) + sizeof(int))) / (1024.0 * 1024.0 * 1024.0);
    std::cout << "  Estimated dense matrix: " << std::fixed << std::setprecision(1) << estimated_dense_gb << " GB" << std::endl;
    std::cout << "  Estimated sparse matrix (1000 nnz/row): " << estimated_sparse_gb << " GB" << std::endl;
    std::cout << "  H-matrix compression target: ~" << (estimated_sparse_gb / 10.0) << " GB" << std::endl;

    // PHASE 1: Precompute face data
    auto phase1_start = std::chrono::high_resolution_clock::now();

    face_centers_.resize(n_faces);
    face_normals_.resize(n_faces);
    face_areas_.resize(n_faces);

    // Recenter geometry to its bounding-box center. A surface patch can sit at
    // planetary-scale absolute coordinates (e.g. lunar selenocentric, |Z|~1.7e6 m)
    // while spanning only meters; at that magnitude a float32 ulp (~0.1 m) exceeds
    // the feature size, collapsing Embree's BVH and making visibility ray tracing
    // pathologically slow. The ray origins below derive from face_centers_, and
    // RadiationExchangeManager applies this SAME bbox-center shift to the Embree
    // scene vertices (computed identically from mesh_->vertices), so rays and scene
    // stay in one small-coordinate space. View factors depend only on relative
    // geometry, so the constant shift is exact; normals/areas use vertex
    // differences and are unaffected.
    const auto& V0 = mesh_->vertices;
    std::array<double,3> bbmin = {{ V0[0][0], V0[0][1], V0[0][2] }};
    std::array<double,3> bbmax = bbmin;
    for (size_t i = 1; i < V0.size(); ++i) {
        for (int k = 0; k < 3; ++k) {
            bbmin[k] = std::min(bbmin[k], V0[i][k]);
            bbmax[k] = std::max(bbmax[k], V0[i][k]);
        }
    }
    const std::array<double,3> origin_shift = {{
        0.5 * (bbmin[0] + bbmax[0]),
        0.5 * (bbmin[1] + bbmax[1]),
        0.5 * (bbmin[2] + bbmax[2])
    }};
    // Published so the caller can CHECK it against the shift it actually
    // applied to the scene, rather than trusting that two copies of the same
    // formula stay in step. They did not, for the sparse builder, for a month.
    scene_origin_shift_ = origin_shift;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n_faces; ++i) {
        const auto& face = mesh_->faces[i];
        const auto& v0 = mesh_->vertices[face[0]];
        const auto& v1 = mesh_->vertices[face[1]];
        const auto& v2 = mesh_->vertices[face[2]];

        // Compute center (in recentered coordinates; see origin_shift above)
        face_centers_[i] = {{
            (v0[0] + v1[0] + v2[0]) / 3.0 - origin_shift[0],
            (v0[1] + v1[1] + v2[1]) / 3.0 - origin_shift[1],
            (v0[2] + v1[2] + v2[2]) / 3.0 - origin_shift[2]
        }};

        // Compute normal and area
        auto e1 = subtract(v1, v0);
        auto e2 = subtract(v2, v0);
        auto n = cross(e1, e2);
        double area2 = length(n);
        face_areas_[i] = area2 / 2.0;

        if (area2 > 0) {
            face_normals_[i] = {{n[0] / area2, n[1] / area2, n[2] / area2}};
        } else {
            face_normals_[i] = {{0.0, 0.0, 1.0}};
        }
    }

    // Optional: heightfield-horizon visibility (robust grazing-tangent occlusion).
    // up = radial = normalize(origin_shift) (the patch's selenocentric center);
    // cell ~ median face pitch (sqrt of mean area). Used in place of the Embree
    // thin-ray in computeViewFactorRow/Column/Block when enabled.
    {
        const char* hv = std::getenv("PTM_HMATRIX_HORIZON_VIS");
        use_horizon_vis_ = (hv && std::atoi(hv) != 0);
        if (use_horizon_vis_) {
            double mean_area = 0.0;
            for (size_t i = 0; i < n_faces; ++i) mean_area += face_areas_[i];
            mean_area = n_faces ? mean_area / n_faces : 1.0;
            double cell = std::sqrt(std::max(1e-6, 2.0 * mean_area));   // ~ face pitch
            std::array<double,3> up = origin_shift;
            double Lu = std::sqrt(up[0]*up[0]+up[1]*up[1]+up[2]*up[2]);
            if (Lu > 0) { up[0]/=Lu; up[1]/=Lu; up[2]/=Lu; } else { up = {{0,0,1}}; }
            std::cout << "  [horizon-vis] building heightfield (cell=" << cell
                      << " m, up=" << up[0] << "," << up[1] << "," << up[2] << ")" << std::endl;
            horizon_.build(face_centers_, up, cell);
        }
    }

    // Optional: smoothed (macroscopic) normals for the VF cosine terms. Raw
    // per-facet normals on fine rough meshes decouple grazing floor<->rim pairs;
    // the local-mean normal gives the resolution-independent VF (validated by
    // bin/vf_hemisphere_mc: floor<->warm 0.0032 macroscopic vs ~0 per-facet at 400k).
    {
        const char* sn = std::getenv("PTM_HMATRIX_SMOOTH_NORMALS");
        use_smoothed_normals_ = (sn && std::atoi(sn) != 0);
        if (use_smoothed_normals_) {
            double R = 300.0;
            if (const char* sr = std::getenv("PTM_HMATRIX_SMOOTH_RADIUS")) {
                double v = std::atof(sr); if (v > 0) R = v;
            }
            std::cout << "  [smooth-normals] computing macroscopic VF normals (radius="
                      << R << " m)" << std::endl;
            computeSmoothedNormals(R);
        }
    }

    auto phase1_end = std::chrono::high_resolution_clock::now();
    double phase1_ms = std::chrono::duration<double, std::milli>(phase1_end - phase1_start).count();

    // PHASE 2: Build spatial partitioning tree (octree for 3D, or quadtree for 2D)
    auto phase2_start = std::chrono::high_resolution_clock::now();

    // Use 3D octree by default for crater geometry (Z variation matters)
    // Octree provides better clustering than quadtree for crater floors/rims
    use_octree_ = true;

    if (config_.verbose) {
        std::cout << "  Building " << (use_octree_ ? "octree (3D)" : "quadtree (2D)") << "..." << std::endl;
    } else {
        std::cout << "  Building " << (use_octree_ ? "octree (3D)" : "quadtree (2D)") << "..." << std::endl;
    }

    std::vector<int> all_indices(n_faces);
    std::iota(all_indices.begin(), all_indices.end(), 0);

    // Target faces per node depends on compression method
    // WITH ACA: Per-block computation is O(n * k) where k is the rank (typically 10-50)
    //   - However, when ACA fails, we fall back to sparse computation
    //   - For very large meshes, we need smaller blocks to:
    //     (a) Increase ACA success rate (better admissibility at smaller scales)
    //     (b) Reduce memory when sparse fallback is needed
    //   - Target ~2000 faces/node for large meshes, ~4000 for medium
    //
    // WITHOUT ACA (SVD): Per-block computation is O(n²)
    //   - Need smaller blocks to keep computation tractable
    //   - Target ~1000 faces/node
    size_t target_faces_per_node;
    if (config_.compression_method == CompressionMethod::ACA) {
        if (n_faces > 2000000) {
            target_faces_per_node = 2000;  // Very large mesh: smaller blocks for better ACA
        } else if (n_faces > 500000) {
            target_faces_per_node = 3000;  // Large mesh
        } else {
            target_faces_per_node = 5000;  // Medium mesh
        }
    } else {
        target_faces_per_node = 1000;  // SVD is O(n²), keep blocks small
    }

    // Calculate target node count based on mesh size
    size_t target_nodes = std::max(size_t(16), (n_faces + target_faces_per_node - 1) / target_faces_per_node);

    // Hard limits - keep block count reasonable
    // For octree at depth d, we have up to 8^d nodes (vs 4^d for quadtree)
    // This means octree can achieve similar node counts at lower depth
    //
    // CRITICAL: Block pairs = nodes², so node count must be limited aggressively
    // - 400 nodes → 160K block pairs (manageable)
    // - 600 nodes → 360K block pairs (heavy)
    // - 1000 nodes → 1M block pairs (too many for very large meshes)
    //
    // For very large meshes (>1M faces), we prioritize completing over optimal compression
    size_t MAX_NODES;
    if (n_faces > 1500000) {
        MAX_NODES = 400;  // 160K block pairs max for very large meshes (2M+ faces)
    } else if (n_faces > 500000) {
        MAX_NODES = 600;  // 360K block pairs max for large meshes
    } else {
        MAX_NODES = 1000; // 1M block pairs max for medium meshes
    }
    const size_t MIN_NODES = 16;        // At least 16 nodes for any mesh

    target_nodes = std::min(target_nodes, MAX_NODES);
    target_nodes = std::max(target_nodes, MIN_NODES);

    if (n_faces > 1000000) {
        std::cout << "  Large mesh optimization:" << std::endl;
        std::cout << "    Compression method: " << (config_.compression_method == CompressionMethod::ACA ? "ACA" : "SVD") << std::endl;
        std::cout << "    Spatial tree: " << (use_octree_ ? "3D octree" : "2D quadtree") << std::endl;
        std::cout << "    Target ~" << target_faces_per_node << " faces/node" << std::endl;
        std::cout << "    Max nodes: " << MAX_NODES << " (~" << MAX_NODES * MAX_NODES / 1000 << "K block pairs)" << std::endl;
    }

    int start_depth = 2;  // Default for small meshes
    size_t num_nodes = 0;

    if (use_octree_) {
        // Build 3D octree
        octree_root_ = std::make_unique<OctreeNode>();
        octree_root_->build(face_centers_, all_indices,
                            config_.max_tree_depth, config_.min_block_size);

        // For octree: 8^d nodes at depth d
        // We want: 8^d ≈ target_nodes, so d ≈ log8(target_nodes)
        if (n_faces > 100000) {
            int target_depth = static_cast<int>(std::ceil(std::log(double(target_nodes)) / std::log(8.0)));
            start_depth = std::max(2, std::min(target_depth, config_.max_tree_depth - 2));

            if (n_faces > 1000000) {
                std::cout << "  Using octree depth " << start_depth
                          << " (targeting ~" << target_nodes << " nodes)" << std::endl;
            }
        }

        std::vector<OctreeNode*> octree_nodes;
        octree_root_->getNodesAtDepth(start_depth, octree_nodes);

        // Adjust depth to get target node count
        while (octree_nodes.size() < target_nodes / 2 && start_depth < config_.max_tree_depth - 2) {
            octree_nodes.clear();
            start_depth++;
            octree_root_->getNodesAtDepth(start_depth, octree_nodes);
            if (n_faces > 1000000) {
                std::cout << "  Increased depth to " << start_depth
                          << " (" << octree_nodes.size() << " nodes)" << std::endl;
            }
        }

        while (octree_nodes.size() > MAX_NODES && start_depth > 1) {
            octree_nodes.clear();
            start_depth--;
            octree_root_->getNodesAtDepth(start_depth, octree_nodes);
            std::cout << "  Reduced depth to " << start_depth << " (" << octree_nodes.size() << " nodes)" << std::endl;
        }

        num_nodes = octree_nodes.size();
    } else {
        // Build 2D quadtree (legacy path)
        quadtree_root_ = std::make_unique<QuadtreeNode>();
        quadtree_root_->build(face_centers_, all_indices,
                              config_.max_tree_depth, config_.min_block_size);

        // For quadtree: 4^d nodes at depth d
        if (n_faces > 100000) {
            int target_depth = static_cast<int>(std::ceil(std::log(double(target_nodes)) / std::log(4.0)));
            start_depth = std::max(2, std::min(target_depth, config_.max_tree_depth - 2));

            if (n_faces > 1000000) {
                std::cout << "  Using quadtree depth " << start_depth
                          << " (targeting ~" << target_nodes << " nodes)" << std::endl;
            }
        }

        std::vector<QuadtreeNode*> quadtree_nodes;
        quadtree_root_->getNodesAtDepth(start_depth, quadtree_nodes);

        while (quadtree_nodes.size() < target_nodes / 2 && start_depth < config_.max_tree_depth - 2) {
            quadtree_nodes.clear();
            start_depth++;
            quadtree_root_->getNodesAtDepth(start_depth, quadtree_nodes);
            if (n_faces > 1000000) {
                std::cout << "  Increased depth to " << start_depth
                          << " (" << quadtree_nodes.size() << " nodes)" << std::endl;
            }
        }

        while (quadtree_nodes.size() > MAX_NODES && start_depth > 1) {
            quadtree_nodes.clear();
            start_depth--;
            quadtree_root_->getNodesAtDepth(start_depth, quadtree_nodes);
            std::cout << "  Reduced depth to " << start_depth << " (" << quadtree_nodes.size() << " nodes)" << std::endl;
        }

        num_nodes = quadtree_nodes.size();
    }

    auto phase2_end = std::chrono::high_resolution_clock::now();
    double phase2_ms = std::chrono::duration<double, std::milli>(phase2_end - phase2_start).count();

    std::cout << "  " << (use_octree_ ? "Octree" : "Quadtree") << ": using depth " << start_depth
              << " (" << num_nodes << " nodes, ~" << (n_faces / std::max(num_nodes, size_t(1))) << " faces/node)" << std::endl;

    // Store starting depth for hierarchical level limiting
    start_depth_ = start_depth;

    // Reset statistics
    num_zero_blocks_ = 0;
    num_sparse_blocks_ = 0;
    num_lowrank_blocks_ = 0;
    num_dense_blocks_ = 0;
    num_hierarchical_blocks_ = 0;
    total_nnz_ = 0;

    // PHASE 3: Build H-matrix blocks for all node pairs
    auto phase3_start = std::chrono::high_resolution_clock::now();

    // Get the node list based on tree type (need it for block building)
    std::vector<OctreeNode*> octree_nodes;
    std::vector<QuadtreeNode*> quadtree_nodes;
    if (use_octree_) {
        octree_root_->getNodesAtDepth(start_depth, octree_nodes);
    } else {
        quadtree_root_->getNodesAtDepth(start_depth, quadtree_nodes);
    }

    size_t total_blocks = num_nodes * num_nodes;

    // Safety check: ensure block count is reasonable
    // With MAX_NODES=2000, we could have up to 4M blocks, but most will be zero
    constexpr size_t MAX_BLOCKS = 4000000;  // 4M blocks max
    if (total_blocks > MAX_BLOCKS) {
        std::cerr << "ERROR: H-matrix would require " << total_blocks << " blocks "
                  << "(max allowed: " << MAX_BLOCKS << ").\n"
                  << "Mesh has " << n_faces << " faces. Consider:\n"
                  << "  - Reducing mesh resolution\n"
                  << "  - Increasing H-matrix tolerance (--rad-hmatrix-tol)\n"
                  << "  - Using a coarser spatial discretization\n";
        throw std::runtime_error("H-matrix too large - block count exceeds safe limit");
    }

    // Memory estimate for block storage
    // Each block: unique_ptr (8) + variant (~64) + row/col indices (~16 per face avg)
    size_t estimated_min_memory = total_blocks * 80;  // Very conservative minimum

    // More realistic memory estimate including index storage
    // Each block pair copies row_indices and col_indices from both nodes
    // For depth-3 octree with 283 nodes and 11,762 faces/node:
    //   - Each block stores ~11,762 × 2 = 23,524 ints = 94 KB per block
    //   - 80,089 blocks × 94 KB = 7.5 GB just for indices!
    size_t faces_per_node = n_faces / std::max(num_nodes, size_t(1));
    size_t index_memory_per_block = faces_per_node * 2 * sizeof(int);
    size_t estimated_index_memory = total_blocks * index_memory_per_block;

    std::cout << "  Building " << total_blocks << " block pairs:" << std::endl;
    std::cout << "    Minimum block overhead: ~" << estimated_min_memory / (1024*1024) << " MB" << std::endl;
    std::cout << "    Estimated index storage: ~" << std::fixed << std::setprecision(1)
              << estimated_index_memory / (1024.0 * 1024.0 * 1024.0) << " GB "
              << "(" << faces_per_node << " faces/node × 2 × " << total_blocks << " blocks)" << std::endl;

    // Initial memory report
    size_t avail_before = getAvailableMemory();
    size_t total_mem = getTotalMemory();
    std::cout << "    System memory: " << std::fixed << std::setprecision(1)
              << avail_before / (1024.0 * 1024.0 * 1024.0) << " GB available / "
              << total_mem / (1024.0 * 1024.0 * 1024.0) << " GB total" << std::endl;

    // Warning if estimated memory exceeds available
    if (estimated_index_memory > avail_before * 0.5) {
        std::cout << "    WARNING: Estimated index memory (" << estimated_index_memory / (1024*1024*1024)
                  << " GB) exceeds 50% of available memory!" << std::endl;
        std::cout << "    Consider increasing octree depth to reduce faces per node." << std::endl;
    }

    // Auto-enable out-of-core if memory estimate exceeds threshold
    // This writes block data to disk as it's computed, keeping only metadata in RAM
    //
    // Memory estimate needs to include ACTUAL block data, not just indices:
    // - Sparse blocks: ~12 bytes per nnz (value + col_idx) + row_ptrs
    // - Low-rank blocks: (n_rows + n_cols) * rank * 8 bytes for U/V + rank * 8 for sigma
    //
    // Empirical observation: For large meshes with ~2000 faces/node and ACA,
    // actual memory is approximately 3-5x the index storage estimate due to:
    // - Low-rank blocks dominate memory (U/V matrices at ~1.6 MB per block with rank 50)
    // - ~5% of blocks become low-rank, ~12% become sparse
    // Conservative multiplier to trigger out-of-core appropriately
    size_t estimated_block_data = estimated_index_memory * 4;  // 4x for actual block data
    size_t estimated_total_memory = estimated_index_memory + estimated_min_memory + estimated_block_data;

    std::cout << "    Estimated total memory (with block data): ~" << std::fixed << std::setprecision(1)
              << estimated_total_memory / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;

    bool auto_enable_out_of_core = false;

    if (config_.memory_limit_gb > 0 &&
        estimated_total_memory > config_.memory_limit_gb * 1024ULL * 1024ULL * 1024ULL) {
        auto_enable_out_of_core = true;
        std::cout << "    Auto-enabling out-of-core storage (estimate exceeds "
                  << config_.memory_limit_gb << " GB limit)" << std::endl;
    }

    // Use lower threshold (50%) for large meshes to avoid late-stage OOM
    // Memory estimates are often too low because they don't account for sparse fallbacks
    double ooc_threshold = (n_faces > 500000) ? 0.5 : 0.8;
    if (estimated_total_memory > avail_before * ooc_threshold) {
        auto_enable_out_of_core = true;
        std::cout << "    Auto-enabling out-of-core storage (estimate exceeds "
                  << static_cast<int>(ooc_threshold * 100) << "% of available RAM)"
                  << std::endl;
    }

    // Force out-of-core for very large meshes regardless of memory estimate
    // because sparse fallback blocks (30-40% of total) use far more memory than estimated
    if (n_faces > 1000000 && !config_.use_out_of_core && !auto_enable_out_of_core) {
        auto_enable_out_of_core = true;
        std::cout << "    Auto-enabling out-of-core storage (mesh > 1M faces)" << std::endl;
    }

    if (config_.use_out_of_core || auto_enable_out_of_core) {
        std::string ooc_path = config_.out_of_core_path;
        if (ooc_path.empty()) {
            // Scratch location: honor $TMPDIR (POSIX; /tmp may be unwritable or
            // small on shared hosts), fall back to /tmp.
            const char* tmpdir = std::getenv("TMPDIR");
            std::string tmp_base = (tmpdir && *tmpdir) ? tmpdir : "/tmp";
            while (tmp_base.size() > 1 && tmp_base.back() == '/') tmp_base.pop_back();
            ooc_path = tmp_base + "/hmatrix_" + std::to_string(getpid());
        }
        if (!initOutOfCore(ooc_path)) {
            std::cerr << "WARNING: Failed to initialize out-of-core storage, continuing in-memory" << std::endl;
        }
    }

    std::cout << std::flush;

    root_blocks_.resize(total_blocks);

    // Progress tracking
    std::atomic<size_t> blocks_completed{0};
    const size_t progress_interval = std::max(total_blocks / 100, static_cast<size_t>(1));

    // CRITICAL: Limit concurrency for large meshes to avoid OOM
    // Each block computation allocates temporary memory for:
    //   - row_candidates: avg_faces_per_node vectors, each potentially hundreds of entries
    //   - all_candidates: flattened version, potentially millions of entries
    //   - ray_visible: boolean array for all candidates
    //   - Various intermediate vectors during visibility testing
    //
    // Memory per block estimate depends on compression method:
    //
    // SVD method (original):
    //   - For N faces/node with M col_faces: up to N*M candidate pairs to evaluate
    //   - Each candidate: ~64 bytes for RayCandidate struct
    //   - After geometric culling, typically ~10-30% of pairs remain as candidates
    //   - For 13K x 13K block: ~170M pairs, ~30M candidates after culling = ~2GB just for candidates
    //   - Conservative estimate: 200 KB per face in the larger dimension
    //
    // ACA method (new):
    //   - Only computes one row OR one column at a time, not the full N×M block
    //   - Memory for row/column: O(N) candidates, not O(N²)
    //   - Each row/column computation: N candidates × 64 bytes = ~640 KB for 10K faces
    //   - Plus rank×N storage for U,V matrices (rank typically 10-50)
    //   - Conservative estimate: 10 KB per face (50× less than SVD)
    //
    size_t avg_faces_per_node = n_faces / std::max(num_nodes, size_t(1));
    int max_threads = omp_get_max_threads();
    int threads_to_use = max_threads;

    // Query available system memory
    size_t available_memory = getAvailableMemory();
    size_t total_memory = getTotalMemory();

    // Estimate memory per concurrent block computation
    // NOTE: ACA is O(n) for successful compression, but ~15-20% of blocks fall back
    // to sparse computation which is O(n²). We use a weighted estimate to be safe.
    size_t mem_per_block;
    if (config_.compression_method == CompressionMethod::ACA) {
        // ACA is O(n), but sparse fallback is O(n²)
        // Sparse fallback: n faces × n faces × ~56 bytes (RayCandidate) / n ≈ 60 KB/face
        // Weighted estimate: 80% ACA (10 KB) + 20% sparse (60 KB) = 20 KB/face
        mem_per_block = avg_faces_per_node * 20ULL * 1024ULL;  // 20 KB per face (conservative)
    } else {
        // SVD: O(n²) memory per block (full candidate matrix)
        // 200 KB per face is conservative estimate
        mem_per_block = avg_faces_per_node * 200ULL * 1024ULL;  // 200 KB per face
    }

    std::cout << "  Thread limiting for " << avg_faces_per_node << " faces/node "
              << "(method: " << (config_.compression_method == CompressionMethod::ACA ? "ACA" : "SVD") << "):" << std::endl;
    std::cout << "    Estimated memory/block: ~" << (mem_per_block / (1024*1024)) << " MB" << std::endl;

    if (available_memory > 0 && total_memory > 0) {
        std::cout << "    System memory: " << (available_memory / (1024*1024*1024)) << " GB available, "
                  << (total_memory / (1024*1024*1024)) << " GB total" << std::endl;

        // Reserve 50% of available memory for safety (system + existing allocations)
        // The illumination data alone uses ~60GB, so available memory may be very low
        size_t usable_memory = available_memory / 2;

        // Calculate safe thread count
        int safe_threads = static_cast<int>(usable_memory / mem_per_block);
        safe_threads = std::max(1, std::min(safe_threads, max_threads));
        threads_to_use = safe_threads;

        std::cout << "    Memory-based limit: " << threads_to_use << " threads "
                  << "(max was " << max_threads << ")" << std::endl;
    } else {
        std::cout << "    Memory query failed, using heuristic" << std::endl;
    }

    // Apply hard limits based on block size regardless of memory query
    // These are safety limits to prevent OOM even if memory reporting is inaccurate
    // ACA uses much less memory, so hard limits are relaxed significantly
    int hard_limit = max_threads;
    if (config_.compression_method == CompressionMethod::ACA) {
        // ACA: O(n) memory for successful compression, but ~30-40% of blocks
        // fall back to sparse computation which uses O(n²) memory per thread.
        // With out-of-core enabled, completed blocks are written to disk, so
        // we can use more threads for faster computation.
        // Working memory per thread is ~200-500 MB for sparse fallback blocks.

        // Thread limits with out-of-core enabled (completed blocks go to disk):
        // CRITICAL: For very large meshes with large blocks (>5000 faces/node),
        // each thread can use 500MB+ during ACA row/column computation.
        // Limit threads aggressively based on faces per node.
        if (out_of_core_enabled_) {
            // Large-block builds blow up memory super-linearly in thread count.
            // Measured on moon3d (1.7M-face bowl crater, 5756 faces/node, OOC):
            // 4 threads drained 116 GB by 21% of the block build and OOM-killed
            // (jetsam, exit -9); 1 thread stayed essentially FLAT (-1.3 GB over
            // the same span). The cause is not the per-block O(n²) working set
            // itself (~tens of MB) but multithreaded allocator fragmentation:
            // each worker's arena grows and freed blocks are not reused across
            // threads, so RSS climbs without bound. The old "~500 MB/thread"
            // estimate undercounted this by ~250x. For sparse-fallback-heavy
            // (concave) clusters we therefore serialize the build; benign
            // (convex) large meshes that compress cleanly can raise the count
            // via PTM_HMATRIX_THREADS (env override applied below).
            if (avg_faces_per_node > 8000 || n_faces > 2500000) {
                hard_limit = std::min(hard_limit, 1);
            } else if (avg_faces_per_node > 5000) {
                // moon3d-class: proven OOM at 4 threads, flat at 1.
                hard_limit = std::min(hard_limit, 1);
            } else if (avg_faces_per_node > 3000) {
                hard_limit = std::min(hard_limit, 2);
            } else if (n_faces > 1500000) {
                hard_limit = std::min(hard_limit, 8);
            } else if (n_faces > 1000000) {
                hard_limit = std::min(hard_limit, 12);
            } else if (n_faces > 500000) {
                hard_limit = std::min(hard_limit, 16);
            }
        } else {
            // Without out-of-core, be very conservative (completed blocks stay in RAM)
            if (n_faces > 1500000) {
                hard_limit = std::min(hard_limit, 2);
            } else if (n_faces > 1000000) {
                hard_limit = std::min(hard_limit, 4);
            } else if (n_faces > 500000) {
                hard_limit = std::min(hard_limit, 6);
            }
        }
    } else {
        // SVD: O(n²) memory, keep original conservative limits
        if (avg_faces_per_node > 10000) {
            // Very large blocks (>10K faces): limit to 2 threads
            // Each block can easily use 2-4 GB during computation
            hard_limit = 2;
        } else if (avg_faces_per_node > 5000) {
            // Large blocks (5K-10K faces): limit to 4 threads
            hard_limit = 4;
        } else if (avg_faces_per_node > 2000) {
            // Medium blocks (2K-5K faces): limit to 8 threads
            hard_limit = 8;
        }
    }

    if (hard_limit < threads_to_use) {
        std::cout << "    Hard limit applied: " << hard_limit << " threads "
                  << "(block size safety)" << std::endl;
        threads_to_use = hard_limit;
    }

    // Apply manual thread limit if specified via config
    if (config_.hmatrix_threads > 0 && config_.hmatrix_threads < threads_to_use) {
        std::cout << "    Manual thread limit: " << config_.hmatrix_threads << " threads" << std::endl;
        threads_to_use = config_.hmatrix_threads;
    }

    // Env override (highest priority): tuning lever for OOM mitigation.
    // Peak build memory scales ~linearly with concurrent block computations,
    // so capping threads is the most direct way to bound RSS on a fixed machine.
    if (const char* env_threads = std::getenv("PTM_HMATRIX_THREADS")) {
        int t = std::atoi(env_threads);
        if (t > 0) {
            std::cout << "    Env PTM_HMATRIX_THREADS=" << t
                      << " (overrides " << threads_to_use << ")" << std::endl;
            threads_to_use = t;
        }
    }

    std::cout << "    Final thread count: " << threads_to_use << std::endl;
    omp_set_num_threads(threads_to_use);
    hmatrix_build_threads_ = threads_to_use;
    std::cout << std::flush;

    // Time estimation for progress reporting
    auto block_start_time = std::chrono::high_resolution_clock::now();
    std::atomic<double> estimated_remaining_sec{-1.0};

    // Memory reporting interval (every 10% or every 5000 blocks)
    size_t memory_report_interval = std::max(total_blocks / 10, static_cast<size_t>(5000));
    size_t last_memory_report = 0;

    // Build blocks using appropriate tree type
    if (use_octree_) {
        #pragma omp parallel for schedule(dynamic) collapse(2)
        for (size_t i = 0; i < octree_nodes.size(); ++i) {
            for (size_t j = 0; j < octree_nodes.size(); ++j) {
                size_t block_idx = i * octree_nodes.size() + j;
                root_blocks_[block_idx] = buildBlock(octree_nodes[i], octree_nodes[j], start_depth);

                // Progress update with time estimation
                size_t completed = ++blocks_completed;
                if (completed % progress_interval == 0 || completed == total_blocks) {
                    #pragma omp critical(hmatrix_progress)
                    {
                        auto now = std::chrono::high_resolution_clock::now();
                        double elapsed_sec = std::chrono::duration<double>(now - block_start_time).count();
                        int pct = static_cast<int>(100 * completed / total_blocks);

                        std::string time_str;
                        if (completed > total_blocks / 100 && completed < total_blocks) {
                            double rate = completed / elapsed_sec;
                            double remaining = (total_blocks - completed) / rate;
                            estimated_remaining_sec.store(remaining);

                            if (remaining < 60) {
                                time_str = " (~" + std::to_string(int(remaining)) + "s remaining)";
                            } else if (remaining < 3600) {
                                int mins = int(remaining / 60);
                                int secs = int(remaining) % 60;
                                time_str = " (~" + std::to_string(mins) + "m " + std::to_string(secs) + "s remaining)";
                            } else {
                                int hours = int(remaining / 3600);
                                int mins = (int(remaining) % 3600) / 60;
                                time_str = " (~" + std::to_string(hours) + "h " + std::to_string(mins) + "m remaining)";
                            }
                        }

                        std::cout << "\r  Block build progress: " << pct << "% ("
                                  << completed << "/" << total_blocks << ")" << time_str
                                  << "          " << std::flush;

                        // Periodic memory report
                        if (completed - last_memory_report >= memory_report_interval) {
                            last_memory_report = completed;
                            size_t avail_mem = getAvailableMemory();
                            size_t idx_mem = g_total_index_bytes.load();

                            // Check for block explosion (too many sub-blocks being created)
                            size_t total_blocks_created = num_zero_blocks_ + num_sparse_blocks_ +
                                num_lowrank_blocks_ + num_hierarchical_blocks_ + num_disk_blocks_;
                            size_t expected_blocks_at_completion = (total_blocks_created * total_blocks) / std::max(completed, size_t(1));

                            // Warn if block explosion detected (>5x expected)
                            if (total_blocks_created > 5 * total_blocks && completed < total_blocks / 2) {
                                std::cout << std::endl << "  [WARNING] Block explosion detected! "
                                          << total_blocks_created << " blocks created from "
                                          << completed << " root pairs" << std::endl;
                                std::cout << "  [WARNING] Expected at completion: ~"
                                          << expected_blocks_at_completion << " blocks" << std::endl;
                                std::cout << "  [WARNING] Consider increasing min_block_size or disabling H-matrix" << std::endl;
                            }

                            // Check for memory pressure (<10 GB available)
                            if (avail_mem < 10ULL * 1024 * 1024 * 1024) {
                                std::cout << std::endl << "  [WARNING] Low memory: "
                                          << (avail_mem / (1024.0 * 1024.0 * 1024.0)) << " GB available"
                                          << std::endl;
                            }

                            std::cout << std::endl << "  [MEM] Available: "
                                      << std::fixed << std::setprecision(1)
                                      << (avail_mem / (1024.0 * 1024.0 * 1024.0)) << " GB, "
                                      << "Index storage: " << (idx_mem / (1024.0 * 1024.0)) << " MB"
                                      << std::endl;

                            // Block type counts for diagnostics
                            std::cout << "  [MEM] Blocks: Zero=" << num_zero_blocks_
                                      << ", Sparse=" << num_sparse_blocks_
                                      << ", LowRank=" << num_lowrank_blocks_
                                      << ", Hier=" << num_hierarchical_blocks_
                                      << ", Disk=" << num_disk_blocks_
                                      << std::endl;

                            // Estimate memory usage by block type
                            // ZeroBlock: ~16 bytes each (just n_rows/n_cols)
                            // DiskBlock: ~64 bytes each (metadata only)
                            // HierarchicalBlock: ~64 bytes + children vector overhead
                            // In-memory blocks: varies significantly
                            size_t zero_mem = num_zero_blocks_ * 16;
                            size_t disk_mem = num_disk_blocks_ * 64;
                            size_t hier_mem = num_hierarchical_blocks_ * 256;  // Estimate with children overhead
                            size_t struct_overhead = (num_zero_blocks_ + num_sparse_blocks_ +
                                                      num_lowrank_blocks_ + num_hierarchical_blocks_ +
                                                      num_disk_blocks_) * 56;  // HMatrixBlock base overhead
                            size_t est_overhead = zero_mem + disk_mem + hier_mem + struct_overhead + idx_mem;

                            std::cout << "  [MEM] Est. tree overhead: "
                                      << (est_overhead / (1024.0 * 1024.0)) << " MB"
                                      << ", Hier recursions: " << g_hierarchical_recursions.load()
                                      << std::endl;

                            // Small block stats (key for memory debugging)
                            size_t small_count = g_small_blocks_count.load();
                            size_t small_bytes = g_small_blocks_bytes.load();
                            std::cout << "  [MEM] Small blocks in RAM: " << small_count
                                      << " (" << (small_bytes / (1024.0 * 1024.0)) << " MB)"
                                      << std::endl;

                            // ACA stats
                            std::cout << "  [MEM] ACA: " << g_aca_successes.load() << "/" << g_aca_attempts.load()
                                      << ", Sparse fallback: " << g_sparse_fallback_calls.load()
                                      << ", Skipped: " << g_sparse_fallback_skipped.load()
                                      << std::endl;

                            // Dynamic memory warning when critically low
                            constexpr size_t CRITICAL_MEMORY_GB = 20;
                            if (avail_mem < CRITICAL_MEMORY_GB * 1024ULL * 1024ULL * 1024ULL) {
                                std::cout << "  [MEM] WARNING: Available memory critically low! "
                                          << "Consider using --rad-hmatrix-threads to limit threads "
                                          << "or increasing system memory." << std::endl;
                            }

                            // Note: no periodic buffer-cache flush is needed. On macOS the
                            // out-of-core fd is opened with F_NOCACHE (see initOutOfCore), so
                            // writes go straight to disk and never accumulate in the unified
                            // buffer cache. The old F_FULLFSYNC here did not evict cached pages
                            // (it only forced dirty data to the platter) and was an expensive
                            // no-op for the RAM problem it was meant to solve.
                        }
                    }
                }
            }
        }
    } else {
        #pragma omp parallel for schedule(dynamic) collapse(2)
        for (size_t i = 0; i < quadtree_nodes.size(); ++i) {
            for (size_t j = 0; j < quadtree_nodes.size(); ++j) {
                size_t block_idx = i * quadtree_nodes.size() + j;
                root_blocks_[block_idx] = buildBlock(quadtree_nodes[i], quadtree_nodes[j], start_depth);

                // Progress update with time estimation
                size_t completed = ++blocks_completed;
                if (completed % progress_interval == 0 || completed == total_blocks) {
                    #pragma omp critical(hmatrix_progress)
                    {
                        auto now = std::chrono::high_resolution_clock::now();
                        double elapsed_sec = std::chrono::duration<double>(now - block_start_time).count();
                        int pct = static_cast<int>(100 * completed / total_blocks);

                        std::string time_str;
                        if (completed > total_blocks / 100 && completed < total_blocks) {
                            double rate = completed / elapsed_sec;
                            double remaining = (total_blocks - completed) / rate;
                            estimated_remaining_sec.store(remaining);

                            if (remaining < 60) {
                                time_str = " (~" + std::to_string(int(remaining)) + "s remaining)";
                            } else if (remaining < 3600) {
                                int mins = int(remaining / 60);
                                int secs = int(remaining) % 60;
                                time_str = " (~" + std::to_string(mins) + "m " + std::to_string(secs) + "s remaining)";
                            } else {
                                int hours = int(remaining / 3600);
                                int mins = (int(remaining) % 3600) / 60;
                                time_str = " (~" + std::to_string(hours) + "h " + std::to_string(mins) + "m remaining)";
                            }
                        }

                        std::cout << "\r  Block build progress: " << pct << "% ("
                                  << completed << "/" << total_blocks << ")" << time_str
                                  << "          " << std::flush;
                    }
                }
            }
        }
    }
    std::cout << std::endl;  // Newline after progress

    // Free spatial tree to save memory (~14 GB for 2M faces)
    // The tree is only needed during block building, not for matvec
    if (use_octree_ && octree_root_) {
        octree_root_.reset();
        std::cout << "  Freed octree after block building" << std::endl;
    } else if (quadtree_root_) {
        quadtree_root_.reset();
        std::cout << "  Freed quadtree after block building" << std::endl;
    }

    // Finalize out-of-core storage if enabled
    if (out_of_core_enabled_) {
        finalizeOutOfCore();

        // Helper lambda to recursively update mmap_base in all DiskBlocks
        std::function<void(HMatrixBlock*)> updateMmapBase = [&](HMatrixBlock* block) {
            if (!block) return;

            if (std::holds_alternative<DiskBlock>(block->data)) {
                auto& disk = std::get<DiskBlock>(block->data);
                disk.mmap_base = out_of_core_mmap_;
            } else if (std::holds_alternative<HierarchicalBlock>(block->data)) {
                // Recurse into children of hierarchical blocks
                auto& hier = std::get<HierarchicalBlock>(block->data);
                for (auto& child : hier.children) {
                    updateMmapBase(child.get());
                }
            }
        };

        // Update all DiskBlock pointers to the mmap base (recursively)
        for (auto& block : root_blocks_) {
            updateMmapBase(block.get());
        }

        std::cout << "  Out-of-core: " << num_disk_blocks_ << " blocks written to disk" << std::endl;
    }

    // Restore original thread count
    omp_set_num_threads(max_threads);

    auto phase3_end = std::chrono::high_resolution_clock::now();
    double phase3_ms = std::chrono::duration<double, std::milli>(phase3_end - phase3_start).count();

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end_time - start_time).count();

    // Near-field physical-bound tripwire. Deliberately OUTSIDE the verbose
    // block: a clamped view factor is an energy-conservation event, and it
    // prints nothing when it did not fire.
    reportNearFieldBoundClamps("H-matrix builder",
                               rad_config_.nearfield_quadrature_order);
    reportVisibilityUnresolvable("H-matrix builder");

    // NOT behind verbose. Which entries of this matrix came through the ACA
    // cross generators, and how close those pairs got to the near-field window,
    // is provenance for the physics -- and until 2026-08-23 the cross
    // generators skipped the near field entirely, so "how close did they get"
    // was the difference between a latent defect and an active one. It is one
    // line, and a verbose-gated diagnostic is a diagnostic nobody reads.
    {
        const size_t cp = g_cross_pairs.load();
        std::cout << "  ACA cross-generator entries: " << cp;
        if (cp > 0) {
            std::cout << "; near field (r < 4d): " << g_cross_near_pairs.load()
                      << "; closest approach r/4d = " << std::fixed << std::setprecision(3)
                      << g_cross_min_r_over_4d.load() << std::defaultfloat;
        } else {
            std::cout << " (ACA did not run on this mesh)";
        }
        std::cout << std::endl;
    }

    // NOT behind verbose. Near pairs whose M1 estimate is below the pre-cull
    // floor now reach the refined kernel instead of being deleted on the
    // strength of the estimator the refinement exists to replace. A zero here
    // is a MEASURED zero -- the change was inert on this mesh -- and a non-zero
    // is the population that used to be silently missing.
    std::cout << "  Near pairs that bypassed the M1 pre-cull: "
              << g_precull_near_bypassed.load() << std::endl;

    // NOT behind verbose, and NOT silent. tryCompressToLowRank() clamps its
    // truncation rank at max_svd_rank and returns success even when the
    // requested compression_tolerance was not met. That is a defensible memory
    // bound and an indefensible silence: a block stored at a tolerance nobody
    // asked for is a physics decision made by a constant.
    {
        const size_t capped = g_svd_rank_capped.load();
        if (capped > 0) {
            std::cout << "  WARNING: SVD rank cap BOUND on " << capped << " of "
                      << g_svd_compress_calls.load() << " compressed blocks: "
                      << "compression_tolerance = " << config_.compression_tolerance
                      << " was NOT met there. max_svd_rank = " << config_.max_svd_rank
                      << ", spectrum asked for up to " << g_svd_rank_cap_worst_want.load()
                      << "; worst delivered relative residual "
                      << std::scientific << std::setprecision(3)
                      << g_svd_rank_cap_worst_resid.load() << std::defaultfloat
                      << ". This matrix is COARSER than its configured tolerance; "
                         "set PTM_HMATRIX_RANK_CAP_FATAL=1 to make that a stop "
                         "instead of a warning." << std::endl;
            const char* fatal = std::getenv("PTM_HMATRIX_RANK_CAP_FATAL");
            if (fatal && std::string(fatal) == "1") {
                std::cerr << "\nFATAL: PTM_HMATRIX_RANK_CAP_FATAL=1 and the SVD rank cap "
                             "bound on " << capped << " block(s).\n"
                             "  The H-matrix does not meet its stated compression "
                             "tolerance, so this run refuses to continue on it."
                          << std::endl;
                std::exit(1);
            }
        }
    }

    if (config_.verbose) {
        std::cout << "  H-matrix build completed in " << elapsed << " seconds" << std::endl;
        std::cout << "  Phase timing breakdown:" << std::endl;
        std::cout << "    Face precompute:  " << std::fixed << std::setprecision(2) << phase1_ms << " ms ("
                  << (100.0 * phase1_ms / (elapsed * 1000.0)) << "%)" << std::endl;
        std::cout << "    Quadtree build:   " << phase2_ms << " ms ("
                  << (100.0 * phase2_ms / (elapsed * 1000.0)) << "%)" << std::endl;
        std::cout << "    Block build:      " << phase3_ms << " ms ("
                  << (100.0 * phase3_ms / (elapsed * 1000.0)) << "%)" << std::endl;
        printProfilingCounters(phase3_ms);
        printStatistics();
    }
}

// Helper to check if two clusters can potentially see each other
bool HMatrixViewFactor::canClustersSeeEachOther(
    QuadtreeNode* row_node,
    QuadtreeNode* col_node) const {

    // Compute average normals for each cluster
    std::array<double, 3> avg_normal_row = {{0, 0, 0}};
    std::array<double, 3> avg_center_row = {{0, 0, 0}};
    for (int idx : row_node->face_indices) {
        avg_normal_row[0] += face_normals_[idx][0];
        avg_normal_row[1] += face_normals_[idx][1];
        avg_normal_row[2] += face_normals_[idx][2];
        avg_center_row[0] += face_centers_[idx][0];
        avg_center_row[1] += face_centers_[idx][1];
        avg_center_row[2] += face_centers_[idx][2];
    }
    double n_row = static_cast<double>(row_node->face_indices.size());
    if (n_row > 0) {
        avg_normal_row[0] /= n_row;
        avg_normal_row[1] /= n_row;
        avg_normal_row[2] /= n_row;
        avg_center_row[0] /= n_row;
        avg_center_row[1] /= n_row;
        avg_center_row[2] /= n_row;
    }

    std::array<double, 3> avg_normal_col = {{0, 0, 0}};
    std::array<double, 3> avg_center_col = {{0, 0, 0}};
    for (int idx : col_node->face_indices) {
        avg_normal_col[0] += face_normals_[idx][0];
        avg_normal_col[1] += face_normals_[idx][1];
        avg_normal_col[2] += face_normals_[idx][2];
        avg_center_col[0] += face_centers_[idx][0];
        avg_center_col[1] += face_centers_[idx][1];
        avg_center_col[2] += face_centers_[idx][2];
    }
    double n_col = static_cast<double>(col_node->face_indices.size());
    if (n_col > 0) {
        avg_normal_col[0] /= n_col;
        avg_normal_col[1] /= n_col;
        avg_normal_col[2] /= n_col;
        avg_center_col[0] /= n_col;
        avg_center_col[1] /= n_col;
        avg_center_col[2] /= n_col;
    }

    // Direction from row cluster to col cluster
    auto r_vec = subtract(avg_center_col, avg_center_row);
    double r2 = dot(r_vec, r_vec);
    if (r2 < 1e-20) return true;  // Overlapping clusters - need to check individually

    double r = std::sqrt(r2);
    std::array<double, 3> r_hat = {{r_vec[0] / r, r_vec[1] / r, r_vec[2] / r}};

    // For the row cluster to see the col cluster:
    // avg_normal_row should point toward col (dot > 0)
    // avg_normal_col should point toward row (dot < 0)
    double dot_row = dot(avg_normal_row, r_hat);
    double dot_col = dot(avg_normal_col, r_hat);

    // Use a relaxed threshold since we're using averages
    // If average normals are strongly opposing, skip the block
    // But be conservative - only skip if very clearly back-facing
    static constexpr double CLUSTER_BACKFACE_THRESHOLD = -0.5;

    // If row cluster clearly faces away from col cluster
    if (dot_row < CLUSTER_BACKFACE_THRESHOLD) return false;

    // If col cluster clearly faces away from row cluster
    if (dot_col > -CLUSTER_BACKFACE_THRESHOLD) return false;

    return true;  // Might be visible, need to check
}

// Octree version of canClustersSeeEachOther (identical logic, different node type)
bool HMatrixViewFactor::canClustersSeeEachOther(
    OctreeNode* row_node,
    OctreeNode* col_node) const {

    // Compute average normals for each cluster
    std::array<double, 3> avg_normal_row = {{0, 0, 0}};
    std::array<double, 3> avg_center_row = {{0, 0, 0}};
    for (int idx : row_node->face_indices) {
        avg_normal_row[0] += face_normals_[idx][0];
        avg_normal_row[1] += face_normals_[idx][1];
        avg_normal_row[2] += face_normals_[idx][2];
        avg_center_row[0] += face_centers_[idx][0];
        avg_center_row[1] += face_centers_[idx][1];
        avg_center_row[2] += face_centers_[idx][2];
    }
    double n_row = static_cast<double>(row_node->face_indices.size());
    if (n_row > 0) {
        avg_normal_row[0] /= n_row;
        avg_normal_row[1] /= n_row;
        avg_normal_row[2] /= n_row;
        avg_center_row[0] /= n_row;
        avg_center_row[1] /= n_row;
        avg_center_row[2] /= n_row;
    }

    std::array<double, 3> avg_normal_col = {{0, 0, 0}};
    std::array<double, 3> avg_center_col = {{0, 0, 0}};
    for (int idx : col_node->face_indices) {
        avg_normal_col[0] += face_normals_[idx][0];
        avg_normal_col[1] += face_normals_[idx][1];
        avg_normal_col[2] += face_normals_[idx][2];
        avg_center_col[0] += face_centers_[idx][0];
        avg_center_col[1] += face_centers_[idx][1];
        avg_center_col[2] += face_centers_[idx][2];
    }
    double n_col = static_cast<double>(col_node->face_indices.size());
    if (n_col > 0) {
        avg_normal_col[0] /= n_col;
        avg_normal_col[1] /= n_col;
        avg_normal_col[2] /= n_col;
        avg_center_col[0] /= n_col;
        avg_center_col[1] /= n_col;
        avg_center_col[2] /= n_col;
    }

    // Direction from row cluster to col cluster
    auto r_vec = subtract(avg_center_col, avg_center_row);
    double r2 = dot(r_vec, r_vec);
    if (r2 < 1e-20) return true;  // Overlapping clusters - need to check individually

    double r = std::sqrt(r2);
    std::array<double, 3> r_hat = {{r_vec[0] / r, r_vec[1] / r, r_vec[2] / r}};

    double dot_row = dot(avg_normal_row, r_hat);
    double dot_col = dot(avg_normal_col, r_hat);

    static constexpr double CLUSTER_BACKFACE_THRESHOLD = -0.5;

    if (dot_row < CLUSTER_BACKFACE_THRESHOLD) return false;
    if (dot_col > -CLUSTER_BACKFACE_THRESHOLD) return false;

    return true;
}

std::unique_ptr<HMatrixBlock> HMatrixViewFactor::buildBlock(
    QuadtreeNode* row_node,
    QuadtreeNode* col_node,
    int depth) {

    auto block = std::make_unique<HMatrixBlock>();
    // NOTE: We defer copying indices until we know the block type.
    // ZeroBlocks don't need indices, saving memory for large meshes.

    int n_rows = static_cast<int>(row_node->face_indices.size());
    int n_cols = static_cast<int>(col_node->face_indices.size());

    // Use references to node indices to avoid copying until we know block type
    const std::vector<int>& row_faces = row_node->face_indices;
    const std::vector<int>& col_faces = col_node->face_indices;

    // Check for empty blocks
    if (n_rows == 0 || n_cols == 0) {
        ZeroBlock zero;
        zero.n_rows = n_rows;
        zero.n_cols = n_cols;
        block->data = zero;
        #pragma omp atomic
        num_zero_blocks_++;
        return block;
    }

    // CLUSTER-LEVEL CULLING: Check if clusters can potentially see each other
    // This is a conservative test using average normals - only skips clearly back-facing clusters
    auto cull_start = std::chrono::high_resolution_clock::now();
    bool clusters_visible = canClustersSeeEachOther(row_node, col_node);
    auto cull_end = std::chrono::high_resolution_clock::now();
    g_time_cluster_cull_us += std::chrono::duration_cast<std::chrono::microseconds>(cull_end - cull_start).count();
    g_cluster_cull_calls++;

    if (!clusters_visible) {
        g_cluster_culled++;
        ZeroBlock zero;
        zero.n_rows = n_rows;
        zero.n_cols = n_cols;
        block->data = zero;
        #pragma omp atomic
        num_zero_blocks_++;
        return block;
    }

    // For small blocks, compute directly
    if (n_rows * n_cols <= config_.min_block_size * config_.min_block_size) {
        // Compute sparse matrix
        SparseBlock sparse;
        auto vf_start = std::chrono::high_resolution_clock::now();
        computeViewFactorBlock(row_faces, col_faces, sparse);
        auto vf_end = std::chrono::high_resolution_clock::now();
        g_time_view_factor_us += std::chrono::duration_cast<std::chrono::microseconds>(vf_end - vf_start).count();
        g_view_factor_calls++;

        if (sparse.values.empty()) {
            // No visible faces - don't need indices
            ZeroBlock zero;
            zero.n_rows = n_rows;
            zero.n_cols = n_cols;
            block->data = zero;
            #pragma omp atomic
            num_zero_blocks_++;
        } else {
            // Non-zero small block - track memory usage (atomic variable, no pragma needed)
            g_small_blocks_count++;

            #pragma omp atomic
            total_nnz_ += sparse.values.size();

            // Try low-rank compression
            LowRankBlock lowrank;
            auto svd_start = std::chrono::high_resolution_clock::now();
            bool compressed = tryCompressToLowRank(sparse, lowrank, config_.compression_tolerance);
            auto svd_end = std::chrono::high_resolution_clock::now();
            g_time_svd_us += std::chrono::duration_cast<std::chrono::microseconds>(svd_end - svd_start).count();
            g_svd_calls++;

            // OUT-OF-CORE: For very large meshes, write even small blocks to disk
            if (out_of_core_enabled_) {
                std::vector<int> row_faces_copy(row_faces);
                std::vector<int> col_faces_copy(col_faces);
                DiskBlock disk;
                if (compressed) {
                    disk = writeBlockToDisk(lowrank, row_faces_copy, col_faces_copy);
                } else {
                    disk = writeBlockToDisk(sparse, row_faces_copy, col_faces_copy);
                }
                block->data = disk;
                // Keep parent indices populated. writeBlock() serializes them; if
                // cleared, loaded H-matrix matvec silently skips every disk-backed
                // block (empty-indices guard in HMatrixBlock::matvec) -> zero flux.
                block->row_indices = row_faces;
                block->col_indices = col_faces;
                #pragma omp atomic
                num_disk_blocks_++;
            } else {
                // In-memory: store as before
                block->row_indices = row_faces;
                block->col_indices = col_faces;
                size_t index_bytes = (n_rows + n_cols) * sizeof(int);
                g_total_index_bytes += index_bytes;

                if (compressed) {
                    size_t block_bytes = lowrank.memoryBytes();
                    g_small_blocks_bytes += block_bytes;
                    block->data = std::move(lowrank);
                    #pragma omp atomic
                    num_lowrank_blocks_++;
                } else {
                    size_t block_bytes = sparse.memoryBytes();
                    g_small_blocks_bytes += block_bytes;
                    block->data = std::move(sparse);
                    #pragma omp atomic
                    num_sparse_blocks_++;
                }
            }
        }
        return block;
    }

    // Check admissibility - if clusters are well-separated, try compression
    if (isAdmissible(row_node, col_node)) {

        // Try ACA first if enabled (much faster for large blocks)
        if (config_.compression_method == CompressionMethod::ACA) {
            LowRankBlock lowrank;
            auto aca_start = std::chrono::high_resolution_clock::now();
            bool aca_success = tryCompressToLowRankACA(
                row_faces, col_faces,
                lowrank, config_.aca_tolerance);
            auto aca_end = std::chrono::high_resolution_clock::now();
            g_time_svd_us += std::chrono::duration_cast<std::chrono::microseconds>(aca_end - aca_start).count();
            g_svd_calls++;

            if (aca_success) {
                // OUT-OF-CORE: Write LowRankBlock to disk instead of storing in RAM
                if (out_of_core_enabled_) {
                    std::vector<int> row_faces_copy(row_faces);
                    std::vector<int> col_faces_copy(col_faces);
                    DiskBlock disk = writeBlockToDisk(lowrank, row_faces_copy, col_faces_copy);

                    block->data = disk;
                    // Keep parent indices populated (see zero-flux bug note).
                    block->row_indices = row_faces;
                    block->col_indices = col_faces;

                    #pragma omp atomic
                    num_disk_blocks_++;
                } else {
                    // In-memory: store as before
                    block->row_indices = row_faces;
                    block->col_indices = col_faces;
                    size_t index_bytes = (n_rows + n_cols) * sizeof(int);
                    g_total_index_bytes += index_bytes;

                    block->data = std::move(lowrank);
                    #pragma omp atomic
                    num_lowrank_blocks_++;
                }
                return block;
            }

            // ACA failed on admissible block - DO NOT subdivide!
            // For well-separated clusters, compute directly as sparse or zero.
            // Subdividing admissible blocks causes exponential block explosion.
            constexpr size_t MAX_DIRECT_COMPUTE = 500;
            if (n_rows <= MAX_DIRECT_COMPUTE && n_cols <= MAX_DIRECT_COMPUTE) {
                SparseBlock sparse;
                computeViewFactorBlock(row_faces, col_faces, sparse);

                if (sparse.values.empty()) {
                    ZeroBlock zero;
                    zero.n_rows = n_rows;
                    zero.n_cols = n_cols;
                    block->data = zero;
                    #pragma omp atomic
                    num_zero_blocks_++;
                } else {
                    #pragma omp atomic
                    total_nnz_ += sparse.values.size();

                    // OUT-OF-CORE: Write sparse block to disk
                    if (out_of_core_enabled_) {
                        std::vector<int> row_faces_copy(row_faces);
                        std::vector<int> col_faces_copy(col_faces);
                        DiskBlock disk = writeBlockToDisk(sparse, row_faces_copy, col_faces_copy);
                        block->data = disk;
                        // Keep parent indices populated (see zero-flux bug note).
                        block->row_indices = row_faces;
                        block->col_indices = col_faces;
                        #pragma omp atomic
                        num_disk_blocks_++;
                    } else {
                        block->row_indices = row_faces;
                        block->col_indices = col_faces;
                        g_total_index_bytes += (n_rows + n_cols) * sizeof(int);
                        block->data = std::move(sparse);
                        #pragma omp atomic
                        num_sparse_blocks_++;
                    }
                }
                return block;
            } else {
                // Large admissible block where ACA failed - treat as zero
                // This is the key fix: prevents memory explosion from subdivision
                g_aca_fail_to_zero++;
                ZeroBlock zero;
                zero.n_rows = n_rows;
                zero.n_cols = n_cols;
                block->data = zero;
                #pragma omp atomic
                num_zero_blocks_++;
                return block;
            }
        }

        // Traditional approach (SVD method only): compute full sparse block, then try SVD compression
        // WARNING: This path uses O(n²) memory and should only be used with few threads
        if (config_.compression_method == CompressionMethod::SVD) {
            SparseBlock sparse;
            auto vf_start = std::chrono::high_resolution_clock::now();
            computeViewFactorBlock(row_faces, col_faces, sparse);
            auto vf_end = std::chrono::high_resolution_clock::now();
            g_time_view_factor_us += std::chrono::duration_cast<std::chrono::microseconds>(vf_end - vf_start).count();
            g_view_factor_calls++;

            if (sparse.values.empty()) {
                ZeroBlock zero;
                zero.n_rows = n_rows;
                zero.n_cols = n_cols;
                block->data = zero;
                #pragma omp atomic
                num_zero_blocks_++;
                return block;
            }

            // Non-zero block - copy indices
            block->row_indices = row_faces;
            block->col_indices = col_faces;
            size_t index_bytes = (n_rows + n_cols) * sizeof(int);
            g_total_index_bytes += index_bytes;

            #pragma omp atomic
            total_nnz_ += sparse.values.size();

            LowRankBlock lowrank;
            auto svd_start = std::chrono::high_resolution_clock::now();
            bool compressed = tryCompressToLowRank(sparse, lowrank, config_.compression_tolerance);
            auto svd_end = std::chrono::high_resolution_clock::now();
            g_time_svd_us += std::chrono::duration_cast<std::chrono::microseconds>(svd_end - svd_start).count();
            g_svd_calls++;

            if (compressed) {
                block->data = std::move(lowrank);
                #pragma omp atomic
                num_lowrank_blocks_++;
                return block;
            }

            // Low-rank failed, use sparse
            block->data = std::move(sparse);
            #pragma omp atomic
            num_sparse_blocks_++;
            return block;
        }

        // ACA mode: if ACA failed on an admissible block, fall through to subdivision
        // (don't return here - let the code below try hierarchical subdivision)
    }

    // Not admissible and not small - subdivide if possible
    // Use hierarchical blocks for better scalability on large meshes
    //
    // MEMORY SAFETY: Limit hierarchical recursion depth relative to start_depth_.
    // max_hierarchical_levels limits how many levels of HierarchicalBlock nesting are allowed.
    int hierarchical_depth = depth - start_depth_;
    bool hierarchical_limit_exceeded = hierarchical_depth >= config_.max_hierarchical_levels;

    if (!row_node->is_leaf && !col_node->is_leaf &&
        depth < config_.max_tree_depth && !hierarchical_limit_exceeded) {

        g_hierarchical_recursions++;

        HierarchicalBlock hier;
        hier.n_rows = n_rows;
        hier.n_cols = n_cols;
        // NOTE: HierarchicalBlock.row_indices/col_indices are NOT used by matvec!
        // matvec just passes through to children which have their own indices.
        // So we DON'T copy indices here - this saves massive memory.

        // Count valid children first to check if subdivision is worthwhile
        int num_valid_pairs = 0;
        for (int ri = 0; ri < 4; ++ri) {
            if (!row_node->children[ri]) continue;
            for (int ci = 0; ci < 4; ++ci) {
                if (!col_node->children[ci]) continue;
                num_valid_pairs++;
            }
        }

        // Only subdivide if there are multiple children (otherwise no benefit)
        if (num_valid_pairs >= 2) {
            hier.children.reserve(num_valid_pairs);

            // Build child blocks for all quadrant pairs
            for (int ri = 0; ri < 4; ++ri) {
                if (!row_node->children[ri]) continue;
                for (int ci = 0; ci < 4; ++ci) {
                    if (!col_node->children[ci]) continue;

                    auto child_block = buildBlock(
                        row_node->children[ri].get(),
                        col_node->children[ci].get(),
                        depth + 1);

                    if (child_block) {
                        hier.children.push_back(std::move(child_block));
                    }
                }
            }

            if (!hier.children.empty()) {
                // NOTE: For HierarchicalBlock, we DON'T need block->row_indices/col_indices
                // The matvec for HierarchicalBlock just passes through to children.
                // NOT copying indices here saves massive memory for large meshes!

                block->data = std::move(hier);
                #pragma omp atomic
                num_hierarchical_blocks_++;
                return block;
            }
        }
    }

    // Fall back to sparse computation
    // WARNING: This uses O(n²) memory. For ACA mode with large blocks, this is dangerous.
    // Only do full sparse computation for small blocks.
    //
    // Cap chosen for two regimes:
    //   - Without OOC: tight (300) to keep concurrent-thread working-set bounded.
    //   - With OOC: 5000. Rationale:
    //     * 1.57M-face probe had ~5809 faces/node with cap=2000 silently zeroing
    //       5749/65469 (8.8%) non-zero blocks. Raising to 5000 covers that regime.
    //     * 4M-face Amundsen build at cap=10000 grew to 315 GB RSS at only 11%
    //       completion on a 384 GB machine — sparse-fallback working memory
    //       (O(n²)·56 B) accumulates too fast on deep-crater geometry even with
    //       the thread limiter. 5000 caps working set at ~1.4 GB per block ×
    //       4 threads = 5.6 GB concurrent; completion is safe.
    static const size_t sparse_cap_override_q = [](){
        const char* e = std::getenv("PTM_HMATRIX_SPARSE_CAP");
        return e ? std::strtoul(e, nullptr, 10) : 0UL;
    }();
    const size_t max_sparse_size_q = sparse_cap_override_q ? sparse_cap_override_q
                                                           : (out_of_core_enabled_ ? 5000 : 300);

    if (config_.compression_method == CompressionMethod::ACA &&
        (n_rows > max_sparse_size_q || n_cols > max_sparse_size_q)) {
        // Block is too large for safe O(n²) computation with many threads
        // Store as zero block - this sacrifices accuracy but prevents OOM
        // This should rarely happen if the tree depth and the cap above are sized
        // to let normal cluster pairs fit.
        g_sparse_fallback_skipped++;
        ZeroBlock zero;
        zero.n_rows = n_rows;
        zero.n_cols = n_cols;
        block->data = zero;
        #pragma omp atomic
        num_zero_blocks_++;
        return block;
    }

    g_sparse_fallback_calls++;
    SparseBlock sparse;
    computeViewFactorBlock(row_faces, col_faces, sparse);

    if (sparse.values.empty()) {
        ZeroBlock zero;
        zero.n_rows = n_rows;
        zero.n_cols = n_cols;
        block->data = zero;
        #pragma omp atomic
        num_zero_blocks_++;
    } else {
        // Non-zero sparse block - copy indices
        block->row_indices = row_faces;
        block->col_indices = col_faces;
        size_t index_bytes = (n_rows + n_cols) * sizeof(int);
        g_total_index_bytes += index_bytes;

        #pragma omp atomic
        total_nnz_ += sparse.values.size();
        block->data = std::move(sparse);
        #pragma omp atomic
        num_sparse_blocks_++;
    }

    return block;
}

// ============================================================================
// Bounded recursive subdivision of a LARGE admissible block
// ============================================================================
// Replaces the old "zero the large admissible block" fallback. Bisects the
// FACE-INDEX LISTS directly (not octree children, which may be leaves) by spatial
// median, re-attempts ACA on each compact sub-cluster, and computes small leaves
// as exact sparse -- so crater floor<->wall coupling survives above ~100k faces
// without the 266 GB single-block RSS balloon. Fan-out is 2 per level and the
// indices are PARTITIONED (never duplicated), so total work is LINEAR in block
// size, not the exponential blow-up the old comment feared (that referred to
// subdividing the 8x8 octree children, not the face lists). RSS is bounded at
// every level: no ACA runs on a side > aca_cap and each sparse leaf is
// <= subdiv_leaf_size. Termination is guaranteed -- each level strictly halves
// the larger dimension, so both reach <= leaf in finite steps where the sparse
// leaf branch returns; the absolute depth cap is only a paranoia backstop.
std::unique_ptr<HMatrixBlock> HMatrixViewFactor::buildAdmissibleSubdivided(
    const std::vector<int>& row_faces,
    const std::vector<int>& col_faces,
    int subdiv_depth,
    bool try_aca) {

    const int n_rows = static_cast<int>(row_faces.size());
    const int n_cols = static_cast<int>(col_faces.size());
    const size_t leaf = static_cast<size_t>(std::max(1, config_.subdiv_leaf_size));
    const size_t aca_cap = getAcaCap();

    // Emit one leaf pair: cheap ACA (compact sub-cluster, within cap) else exact
    // sparse (small enough) else nullptr to signal "recurse". Mirrors the existing
    // ACA-success and <=500-sparse paths, honoring out-of-core and the zero-flux
    // index guard. ACA is skipped on the entry block (subdiv_depth == 0): we are
    // here precisely because ACA on the full block is too big or already failed.
    auto emit_leaf = [&](const std::vector<int>& rf,
                         const std::vector<int>& cf) -> std::unique_ptr<HMatrixBlock> {
        const int rn = static_cast<int>(rf.size());
        const int cn = static_cast<int>(cf.size());
        auto blk = std::make_unique<HMatrixBlock>();

        // (a) Compact sub-cluster within the ACA cap: try cheap low-rank first.
        // Skipped for near-field (inadmissible) reroutes (try_aca=false): those
        // blocks are not low-rank, so a rank-capped ACA would under-count the
        // strong floor<->wall coupling and inject small negative entries; we want
        // exact sparse leaves there.
        if (try_aca && subdiv_depth > 0 &&
            (aca_cap == 0 || ((size_t)rn <= aca_cap && (size_t)cn <= aca_cap)) &&
            rn >= config_.aca_min_block_size && cn >= config_.aca_min_block_size) {
            LowRankBlock lowrank;
            auto t0 = std::chrono::high_resolution_clock::now();
            bool ok = tryCompressToLowRankACA(rf, cf, lowrank, config_.aca_tolerance);
            auto t1 = std::chrono::high_resolution_clock::now();
            g_time_svd_us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
            g_svd_calls++;
            if (ok) {
                if (out_of_core_enabled_) {
                    std::vector<int> rc(rf), cc(cf);
                    blk->data = writeBlockToDisk(lowrank, rc, cc);
                    blk->row_indices = rf;
                    blk->col_indices = cf;
                    #pragma omp atomic
                    num_disk_blocks_++;
                } else {
                    blk->row_indices = rf;
                    blk->col_indices = cf;
                    g_total_index_bytes += (rn + cn) * sizeof(int);
                    blk->data = std::move(lowrank);
                    #pragma omp atomic
                    num_lowrank_blocks_++;
                }
                g_subdiv_leaves_lowrank++;
                return blk;
            }
        }

        // (b) Small enough for an exact sparse leaf.
        if ((size_t)rn <= leaf && (size_t)cn <= leaf) {
            SparseBlock sparse;
            computeViewFactorBlock(rf, cf, sparse);
            if (sparse.values.empty()) {
                ZeroBlock zero;
                zero.n_rows = rn;
                zero.n_cols = cn;
                blk->data = zero;
                #pragma omp atomic
                num_zero_blocks_++;
                g_subdiv_leaves_zero++;
            } else {
                #pragma omp atomic
                total_nnz_ += sparse.values.size();
                if (out_of_core_enabled_) {
                    std::vector<int> rc(rf), cc(cf);
                    blk->data = writeBlockToDisk(sparse, rc, cc);
                    blk->row_indices = rf;
                    blk->col_indices = cf;
                    #pragma omp atomic
                    num_disk_blocks_++;
                } else {
                    blk->row_indices = rf;
                    blk->col_indices = cf;
                    g_total_index_bytes += (rn + cn) * sizeof(int);
                    blk->data = std::move(sparse);
                    #pragma omp atomic
                    num_sparse_blocks_++;
                }
                g_subdiv_leaves_sparse++;
            }
            return blk;
        }

        // (c) Still large and ACA didn't take it -> signal "recurse".
        return nullptr;
    };

    // Absolute depth backstop (paranoia only; natural termination is guaranteed).
    // 2^64 faces is absurd, so this never fires for a real mesh; if it ever did,
    // preserve the legacy OOM safety by zeroing the residual with a warning.
    constexpr int kAbsDepthCap = 64;
    if (subdiv_depth >= kAbsDepthCap &&
        ((size_t)n_rows > leaf || (size_t)n_cols > leaf)) {
        static std::atomic<int> cap_logs{0};
        if (cap_logs.fetch_add(1) < 8) {
            #pragma omp critical(hmatrix_progress)
            std::cout << "  [subdiv] WARNING depth cap hit, zeroing residual "
                      << n_rows << "x" << n_cols << " block (last resort)" << std::endl;
        }
        auto blk = std::make_unique<HMatrixBlock>();
        ZeroBlock zero;
        zero.n_rows = n_rows;
        zero.n_cols = n_cols;
        blk->data = zero;
        #pragma omp atomic
        num_zero_blocks_++;
        g_subdiv_depth_cap_zero++;
        g_aca_fail_to_zero++;   // keep the legacy metric meaningful
        return blk;
    }

    // Try to emit THIS pair as a single leaf (cheap ACA or exact sparse).
    if (auto leaf_blk = emit_leaf(row_faces, col_faces)) {
        return leaf_blk;
    }

    // Otherwise spatially bisect the LARGER face list along its longest centroid
    // extent (median split -> balanced halves -> predictable depth) and recurse.
    const bool split_rows = (n_rows >= n_cols);
    const std::vector<int>& src = split_rows ? row_faces : col_faces;

    double lo[3] = { 1e300, 1e300, 1e300 };
    double hi[3] = { -1e300, -1e300, -1e300 };
    for (int f : src) {
        const auto& c = face_centers_[f];
        for (int a = 0; a < 3; ++a) {
            lo[a] = std::min(lo[a], c[a]);
            hi[a] = std::max(hi[a], c[a]);
        }
    }
    int axis = 0;
    double best_ext = hi[0] - lo[0];
    for (int a = 1; a < 3; ++a) {
        double ext = hi[a] - lo[a];
        if (ext > best_ext) { best_ext = ext; axis = a; }
    }

    std::vector<int> ordered(src);
    auto mid = ordered.begin() + ordered.size() / 2;
    std::nth_element(ordered.begin(), mid, ordered.end(),
        [&](int p, int q) { return face_centers_[p][axis] < face_centers_[q][axis]; });
    std::vector<int> a_half(ordered.begin(), mid);
    std::vector<int> b_half(mid, ordered.end());

    // Degenerate guard: coincident centroids defeat the spatial split -> halve by
    // index instead (still strictly reduces size, so termination holds).
    if (a_half.empty() || b_half.empty()) {
        a_half.assign(src.begin(), src.begin() + src.size() / 2);
        b_half.assign(src.begin() + src.size() / 2, src.end());
    }

    auto hier_blk = std::make_unique<HMatrixBlock>();
    HierarchicalBlock hier;
    hier.n_rows = n_rows;
    hier.n_cols = n_cols;
    // HierarchicalBlock carries no own indices (matvec passes globals to children).
    hier.children.reserve(2);
    if (split_rows) {
        hier.children.push_back(buildAdmissibleSubdivided(a_half, col_faces, subdiv_depth + 1, try_aca));
        hier.children.push_back(buildAdmissibleSubdivided(b_half, col_faces, subdiv_depth + 1, try_aca));
    } else {
        hier.children.push_back(buildAdmissibleSubdivided(row_faces, a_half, subdiv_depth + 1, try_aca));
        hier.children.push_back(buildAdmissibleSubdivided(row_faces, b_half, subdiv_depth + 1, try_aca));
    }
    hier_blk->data = std::move(hier);
    #pragma omp atomic
    num_hierarchical_blocks_++;
    return hier_blk;
}

// ============================================================================
// Octree version of buildBlock (3D spatial partitioning)
// ============================================================================

std::unique_ptr<HMatrixBlock> HMatrixViewFactor::buildBlock(
    OctreeNode* row_node,
    OctreeNode* col_node,
    int depth) {

    auto block = std::make_unique<HMatrixBlock>();
    // NOTE: We defer copying indices until we know the block type.
    // ZeroBlocks don't need indices, saving ~7 GB for large meshes.
    // Indices are copied lazily when needed below.

    int n_rows = static_cast<int>(row_node->face_indices.size());
    int n_cols = static_cast<int>(col_node->face_indices.size());

    // Check for empty blocks
    if (n_rows == 0 || n_cols == 0) {
        ZeroBlock zero;
        zero.n_rows = n_rows;
        zero.n_cols = n_cols;
        block->data = zero;
        #pragma omp atomic
        num_zero_blocks_++;
        return block;
    }

    // CLUSTER-LEVEL CULLING: Check if clusters can potentially see each other
    auto cull_start = std::chrono::high_resolution_clock::now();
    bool clusters_visible = canClustersSeeEachOther(row_node, col_node);
    auto cull_end = std::chrono::high_resolution_clock::now();
    g_time_cluster_cull_us += std::chrono::duration_cast<std::chrono::microseconds>(cull_end - cull_start).count();
    g_cluster_cull_calls++;

    if (!clusters_visible) {
        g_cluster_culled++;
        ZeroBlock zero;
        zero.n_rows = n_rows;
        zero.n_cols = n_cols;
        block->data = zero;
        #pragma omp atomic
        num_zero_blocks_++;
        return block;
    }

    // Use references to node indices to avoid copying until we know block type
    const std::vector<int>& row_faces = row_node->face_indices;
    const std::vector<int>& col_faces = col_node->face_indices;

    // For small blocks, compute directly
    if (n_rows * n_cols <= config_.min_block_size * config_.min_block_size) {
        SparseBlock sparse;
        auto vf_start = std::chrono::high_resolution_clock::now();
        computeViewFactorBlock(row_faces, col_faces, sparse);
        auto vf_end = std::chrono::high_resolution_clock::now();
        g_time_view_factor_us += std::chrono::duration_cast<std::chrono::microseconds>(vf_end - vf_start).count();
        g_view_factor_calls++;

        if (sparse.values.empty()) {
            ZeroBlock zero;
            zero.n_rows = n_rows;
            zero.n_cols = n_cols;
            block->data = zero;
            #pragma omp atomic
            num_zero_blocks_++;
        } else {
            // Non-zero small block - track memory usage (atomic variable, no pragma needed)
            g_small_blocks_count++;

            #pragma omp atomic
            total_nnz_ += sparse.values.size();

            LowRankBlock lowrank;
            auto svd_start = std::chrono::high_resolution_clock::now();
            bool compressed = tryCompressToLowRank(sparse, lowrank, config_.compression_tolerance);
            auto svd_end = std::chrono::high_resolution_clock::now();
            g_time_svd_us += std::chrono::duration_cast<std::chrono::microseconds>(svd_end - svd_start).count();
            g_svd_calls++;

            // OUT-OF-CORE: For very large meshes, write even small blocks to disk
            if (out_of_core_enabled_) {
                std::vector<int> row_faces_copy(row_faces);
                std::vector<int> col_faces_copy(col_faces);
                DiskBlock disk;
                if (compressed) {
                    disk = writeBlockToDisk(lowrank, row_faces_copy, col_faces_copy);
                } else {
                    disk = writeBlockToDisk(sparse, row_faces_copy, col_faces_copy);
                }
                block->data = disk;
                // Keep parent indices populated. writeBlock() serializes them; if
                // cleared, loaded H-matrix matvec silently skips every disk-backed
                // block (empty-indices guard in HMatrixBlock::matvec) -> zero flux.
                block->row_indices = row_faces;
                block->col_indices = col_faces;
                #pragma omp atomic
                num_disk_blocks_++;
            } else {
                // In-memory: store as before
                block->row_indices = row_faces;
                block->col_indices = col_faces;
                size_t index_bytes = (n_rows + n_cols) * sizeof(int);
                g_total_index_bytes += index_bytes;

                if (compressed) {
                    size_t block_bytes = lowrank.memoryBytes();
                    g_small_blocks_bytes += block_bytes;
                    block->data = std::move(lowrank);
                    #pragma omp atomic
                    num_lowrank_blocks_++;
                } else {
                    size_t block_bytes = sparse.memoryBytes();
                    g_small_blocks_bytes += block_bytes;
                    block->data = std::move(sparse);
                    #pragma omp atomic
                    num_sparse_blocks_++;
                }
            }
        }
        return block;
    }

    // Check admissibility - if clusters are well-separated (in 3D), try compression
    if (isAdmissible(row_node, col_node)) {

        // Try ACA first if enabled
        if (config_.compression_method == CompressionMethod::ACA) {
            // Guard: skip ACA for pathologically large admissible blocks.
            // A near-coplanar crater floor cannot be split in Z by the 3D octree,
            // so one node can hold hundreds of thousands of faces. ACA on such a
            // block costs aca_max_rank * O(n) ray-traced row/column evaluations
            // and balloons RSS (observed: a single block freezing the build for
            // 10+ min at ~266 GB before the existing aca-fail-to-zero fallback
            // could trigger). Admissible (well-separated) blocks contribute little
            // and the code already zeros large ACA failures, so we zero these up
            // front. Tunable via PTM_HMATRIX_ACA_CAP (faces per side; 0 = off).
            const size_t aca_cap = getAcaCap();
            if (aca_cap && ((size_t)n_rows > aca_cap || (size_t)n_cols > aca_cap)) {
                // Large admissible block: ACA on it would balloon RSS. Instead of
                // zeroing (which decouples crater floor<->wall above ~100k faces),
                // bisect the face-index lists and re-attempt ACA on compact
                // sub-clusters (bounded RSS). OFF reproduces the legacy zero.
                if (subdivideEnabled(config_)) {
                    static std::atomic<int> sub_logs{0};
                    if (sub_logs.fetch_add(1) < 8) {
                        #pragma omp critical(hmatrix_progress)
                        std::cout << "  [subdiv] bisecting large admissible block "
                                  << n_rows << "x" << n_cols
                                  << " (> aca_cap " << aca_cap << ") instead of zeroing"
                                  << std::endl;
                    }
                    g_admissible_subdivided++;
                    return buildAdmissibleSubdivided(row_faces, col_faces, /*subdiv_depth=*/0);
                }
                static std::atomic<int> capped_logs{0};
                if (capped_logs.fetch_add(1) < 8) {
                    #pragma omp critical(hmatrix_progress)
                    std::cout << "  [ACA-cap] zeroed large admissible block "
                              << n_rows << "x" << n_cols
                              << " (> " << aca_cap << "); set PTM_HMATRIX_ACA_CAP to tune"
                              << std::endl;
                }
                g_aca_fail_to_zero++;
                ZeroBlock zero;
                zero.n_rows = n_rows;
                zero.n_cols = n_cols;
                block->data = zero;
                #pragma omp atomic
                num_zero_blocks_++;
                return block;
            }
            LowRankBlock lowrank;
            auto aca_start = std::chrono::high_resolution_clock::now();
            bool aca_success = tryCompressToLowRankACA(
                row_faces, col_faces,
                lowrank, config_.aca_tolerance);
            auto aca_end = std::chrono::high_resolution_clock::now();
            g_time_svd_us += std::chrono::duration_cast<std::chrono::microseconds>(aca_end - aca_start).count();
            g_svd_calls++;

            if (aca_success) {
                // OUT-OF-CORE: Write LowRankBlock to disk instead of storing in RAM
                if (out_of_core_enabled_) {
                    std::vector<int> row_faces_copy(row_faces);
                    std::vector<int> col_faces_copy(col_faces);
                    DiskBlock disk = writeBlockToDisk(lowrank, row_faces_copy, col_faces_copy);

                    block->data = disk;
                    // Keep parent indices populated (see zero-flux bug note).
                    block->row_indices = row_faces;
                    block->col_indices = col_faces;

                    #pragma omp atomic
                    num_disk_blocks_++;
                } else {
                    // In-memory: store as before
                    block->row_indices = row_faces;
                    block->col_indices = col_faces;
                    size_t index_bytes = (n_rows + n_cols) * sizeof(int);
                    g_total_index_bytes += index_bytes;

                    block->data = std::move(lowrank);
                    #pragma omp atomic
                    num_lowrank_blocks_++;
                }
                return block;
            }

            // ACA failed on admissible block - DO NOT subdivide!
            // For well-separated clusters, compute directly as sparse or zero.
            // Subdividing admissible blocks causes exponential block explosion.
            //
            // For large blocks, approximating as zero is acceptable since:
            // 1. They are far apart (admissible = well-separated)
            // 2. View factors fall off as 1/r^2, so contribution is small
            // 3. ACA failure often means the block is nearly zero anyway
            constexpr size_t MAX_DIRECT_COMPUTE = 500;  // Only compute small blocks directly
            if (n_rows <= MAX_DIRECT_COMPUTE && n_cols <= MAX_DIRECT_COMPUTE) {
                // Small enough to compute directly
                SparseBlock sparse;
                computeViewFactorBlock(row_faces, col_faces, sparse);

                if (sparse.values.empty()) {
                    ZeroBlock zero;
                    zero.n_rows = n_rows;
                    zero.n_cols = n_cols;
                    block->data = zero;
                    #pragma omp atomic
                    num_zero_blocks_++;
                } else {
                    #pragma omp atomic
                    total_nnz_ += sparse.values.size();

                    // OUT-OF-CORE: Write sparse block to disk
                    if (out_of_core_enabled_) {
                        std::vector<int> row_faces_copy(row_faces);
                        std::vector<int> col_faces_copy(col_faces);
                        DiskBlock disk = writeBlockToDisk(sparse, row_faces_copy, col_faces_copy);
                        block->data = disk;
                        // Keep parent indices populated (see zero-flux bug note).
                        block->row_indices = row_faces;
                        block->col_indices = col_faces;
                        #pragma omp atomic
                        num_disk_blocks_++;
                    } else {
                        block->row_indices = row_faces;
                        block->col_indices = col_faces;
                        g_total_index_bytes += (n_rows + n_cols) * sizeof(int);
                        block->data = std::move(sparse);
                        #pragma omp atomic
                        num_sparse_blocks_++;
                    }
                }
                return block;
            } else {
                // Large admissible block where ACA failed. Bisect the face-index
                // lists and re-attempt ACA on compact sub-clusters instead of
                // zeroing (which decoupled crater floor<->wall above ~100k faces).
                // OFF reproduces the legacy zero behavior for A/B verification.
                if (subdivideEnabled(config_)) {
                    g_admissible_subdivided++;
                    return buildAdmissibleSubdivided(row_faces, col_faces, /*subdiv_depth=*/0);
                }
                g_aca_fail_to_zero++;
                ZeroBlock zero;
                zero.n_rows = n_rows;
                zero.n_cols = n_cols;
                block->data = zero;
                #pragma omp atomic
                num_zero_blocks_++;
                return block;
            }
        }

        // Traditional SVD approach (only for SVD compression method)
        if (config_.compression_method == CompressionMethod::SVD) {
            SparseBlock sparse;
            auto vf_start = std::chrono::high_resolution_clock::now();
            computeViewFactorBlock(row_faces, col_faces, sparse);
            auto vf_end = std::chrono::high_resolution_clock::now();
            g_time_view_factor_us += std::chrono::duration_cast<std::chrono::microseconds>(vf_end - vf_start).count();
            g_view_factor_calls++;

            if (sparse.values.empty()) {
                ZeroBlock zero;
                zero.n_rows = n_rows;
                zero.n_cols = n_cols;
                block->data = zero;
                #pragma omp atomic
                num_zero_blocks_++;
                return block;
            }

            // Non-zero block - copy indices
            block->row_indices = row_faces;
            block->col_indices = col_faces;
            size_t index_bytes = (n_rows + n_cols) * sizeof(int);
            g_total_index_bytes += index_bytes;

            #pragma omp atomic
            total_nnz_ += sparse.values.size();

            LowRankBlock lowrank;
            auto svd_start = std::chrono::high_resolution_clock::now();
            bool compressed = tryCompressToLowRank(sparse, lowrank, config_.compression_tolerance);
            auto svd_end = std::chrono::high_resolution_clock::now();
            g_time_svd_us += std::chrono::duration_cast<std::chrono::microseconds>(svd_end - svd_start).count();
            g_svd_calls++;

            if (compressed) {
                block->data = std::move(lowrank);
                #pragma omp atomic
                num_lowrank_blocks_++;
                return block;
            }

            block->data = std::move(sparse);
            #pragma omp atomic
            num_sparse_blocks_++;
            return block;
        }
    }

    // Not admissible and not small - subdivide if possible
    // Octree has 8 children instead of quadtree's 4
    //
    // MEMORY SAFETY: Limit hierarchical recursion depth relative to start_depth_.
    // With 64 child pairs per level, deep recursion can create millions of blocks.
    // max_hierarchical_levels limits how many levels of HierarchicalBlock nesting are allowed.
    // For 2M face mesh with start_depth=3 and max_hierarchical_levels=2:
    //   - Depth 3: buildBlock called from top level (allowed)
    //   - Depth 4: first level of hierarchical subdivision (allowed)
    //   - Depth 5: second level of hierarchical subdivision (allowed)
    //   - Depth 6+: blocked, falls through to sparse/zero
    int hierarchical_depth = depth - start_depth_;
    bool hierarchical_limit_exceeded = hierarchical_depth >= config_.max_hierarchical_levels;

    if (!row_node->is_leaf && !col_node->is_leaf &&
        depth < config_.max_tree_depth && !hierarchical_limit_exceeded) {

        g_hierarchical_recursions++;

        HierarchicalBlock hier;
        hier.n_rows = n_rows;
        hier.n_cols = n_cols;
        // NOTE: HierarchicalBlock.row_indices/col_indices are NOT used by matvec!
        // matvec just passes through to children which have their own indices.
        // So we DON'T copy indices here - this saves massive memory.

        // Count valid children (up to 8x8 = 64 pairs for octree)
        int num_valid_pairs = 0;
        for (int ri = 0; ri < 8; ++ri) {
            if (!row_node->children[ri]) continue;
            for (int ci = 0; ci < 8; ++ci) {
                if (!col_node->children[ci]) continue;
                num_valid_pairs++;
            }
        }

        if (num_valid_pairs >= 2) {
            hier.children.reserve(num_valid_pairs);

            // Build child blocks for all octant pairs
            for (int ri = 0; ri < 8; ++ri) {
                if (!row_node->children[ri]) continue;
                for (int ci = 0; ci < 8; ++ci) {
                    if (!col_node->children[ci]) continue;

                    auto child_block = buildBlock(
                        row_node->children[ri].get(),
                        col_node->children[ci].get(),
                        depth + 1);

                    if (child_block) {
                        hier.children.push_back(std::move(child_block));
                    }
                }
            }

            if (!hier.children.empty()) {
                // NOTE: For HierarchicalBlock, we DON'T need block->row_indices/col_indices
                // The matvec for HierarchicalBlock just passes through to children.
                // NOT copying indices here saves ~7 GB for large meshes!

                block->data = std::move(hier);
                #pragma omp atomic
                num_hierarchical_blocks_++;
                return block;
            }
        }
    }

    // Fall back to sparse computation
    // CRITICAL: With 48 threads each potentially computing a sparse block,
    // memory usage per thread = O(n² × sizeof(RayCandidate)) during computation.
    // RayCandidate is ~56 bytes, so 300×300 block = ~5 MB working memory per thread.
    // Keep this limit LOW to prevent 48 × 5 MB = 240 MB per batch of simultaneous blocks.
    //
    // For very large meshes, even this can add up. The blocks that reach here are
    // non-admissible (close together) and couldn't be subdivided further, so they
    // represent genuine nearby interactions that matter for accuracy.
    //
    // OUT-OF-CORE MODE: see the matching quadtree comment for rationale.
    // 5000 is a middle ground: covers a polar rim site's ~5809 avg faces/node, and
    // avoids the 315-GB-at-11% blowup seen on 4M Amundsen at cap=10000.
    // Env PTM_HMATRIX_SPARSE_CAP overrides this for OOM tuning: a smaller cap
    // skips (zeros) larger nearby blocks instead of doing the O(n²) sparse
    // compute, trading some near-field accuracy for bounded working memory.
    static const size_t sparse_cap_override = [](){
        const char* e = std::getenv("PTM_HMATRIX_SPARSE_CAP");
        return e ? std::strtoul(e, nullptr, 10) : 0UL;
    }();
    size_t max_sparse_size = sparse_cap_override ? sparse_cap_override
                                                 : (out_of_core_enabled_ ? 5000 : 300);

    if (config_.compression_method == CompressionMethod::ACA &&
        (n_rows > max_sparse_size || n_cols > max_sparse_size)) {
        // Inadmissible (near-field) block too large for a single sparse compute.
        // This is the strongest, most important coupling (e.g. crater floor <->
        // nearby wall): zeroing it under-radiates PSR floors above ~100k faces,
        // where the coplanar floor cannot be split in Z by the octree and its
        // near-field blocks stay > max_sparse_size after hierarchical subdivision.
        // Bisect the face-index lists and compute bounded sparse leaves instead of
        // zeroing. OFF (or PTM_HMATRIX_SPARSE_CAP) reproduces the legacy zero.
        if (subdivideEnabled(config_)) {
            g_admissible_subdivided++;
            return buildAdmissibleSubdivided(row_faces, col_faces, /*subdiv_depth=*/0,
                                             /*try_aca=*/false);
        }
        g_sparse_fallback_skipped++;
        ZeroBlock zero;
        zero.n_rows = n_rows;
        zero.n_cols = n_cols;
        block->data = zero;
        #pragma omp atomic
        num_zero_blocks_++;
        return block;
    }

    g_sparse_fallback_calls++;
    SparseBlock sparse;
    computeViewFactorBlock(row_faces, col_faces, sparse);

    if (sparse.values.empty()) {
        ZeroBlock zero;
        zero.n_rows = n_rows;
        zero.n_cols = n_cols;
        block->data = zero;
        #pragma omp atomic
        num_zero_blocks_++;
    } else {
        #pragma omp atomic
        total_nnz_ += sparse.values.size();

        // OUT-OF-CORE: Write block to disk instead of storing in RAM
        if (out_of_core_enabled_) {
            // Write sparse data to disk, store only DiskBlock metadata in RAM
            std::vector<int> row_faces_copy(row_faces);
            std::vector<int> col_faces_copy(col_faces);
            DiskBlock disk = writeBlockToDisk(sparse, row_faces_copy, col_faces_copy);

            block->data = disk;
            // Keep parent indices populated (see zero-flux bug note).
            block->row_indices = row_faces;
            block->col_indices = col_faces;

            #pragma omp atomic
            num_disk_blocks_++;
        } else {
            // In-memory: store as before
            block->row_indices = row_faces;
            block->col_indices = col_faces;
            size_t index_bytes = (n_rows + n_cols) * sizeof(int);
            g_total_index_bytes += index_bytes;

            block->data = std::move(sparse);
            #pragma omp atomic
            num_sparse_blocks_++;
        }
    }

    return block;
}

// Structure to hold ray data for batched visibility testing
struct RayCandidate {
    int local_row;      // Local row index in block
    int local_col;      // Local col index in block
    double estimated_F; // Pre-computed view factor (assuming visibility)
    std::array<double, 3> origin;
    std::array<double, 3> dir;
    float tfar;
};

// Runtime detection of AVX-512 support
static bool hasAVX512() {
#if defined(__x86_64__) || defined(_M_X64)
    static bool detected = false;
    static bool has_avx512 = false;
    if (!detected) {
        detected = true;
#if defined(__GNUC__) || defined(__clang__)
        // Use cpuid to check for AVX-512F (bit 16 of EBX for leaf 7)
        unsigned int eax, ebx, ecx, edx;
        __asm__ __volatile__ (
            "cpuid"
            : "=a" (eax), "=b" (ebx), "=c" (ecx), "=d" (edx)
            : "a" (7), "c" (0)
        );
        has_avx512 = (ebx & (1 << 16)) != 0;  // AVX-512F
#elif defined(_MSC_VER)
        int cpuInfo[4];
        __cpuidex(cpuInfo, 7, 0);
        has_avx512 = (cpuInfo[1] & (1 << 16)) != 0;  // AVX-512F
#endif
    }
    return has_avx512;
#else
    return false;
#endif
}

// AVX-512 16-wide ray packet visibility test
static void batchedVisibilityTest16(
    RTCScene scene,
    const std::vector<RayCandidate>& candidates,
    std::vector<bool>& visible,
    size_t start_idx,
    size_t end_idx) {

    constexpr int BATCH_SIZE = 16;

    // 64-byte aligned arrays for AVX-512
    alignas(64) float org_x[BATCH_SIZE];
    alignas(64) float org_y[BATCH_SIZE];
    alignas(64) float org_z[BATCH_SIZE];
    alignas(64) float dir_x[BATCH_SIZE];
    alignas(64) float dir_y[BATCH_SIZE];
    alignas(64) float dir_z[BATCH_SIZE];
    alignas(64) float tnear[BATCH_SIZE];
    alignas(64) float tfar_arr[BATCH_SIZE];
    alignas(64) int valid[BATCH_SIZE];

    size_t i = start_idx;
    while (i < end_idx) {
        int batch_count = 0;
        size_t batch_start = i;

        // Fill batch
        while (batch_count < BATCH_SIZE && i < end_idx) {
            const auto& c = candidates[i];
            org_x[batch_count] = static_cast<float>(c.origin[0]);
            org_y[batch_count] = static_cast<float>(c.origin[1]);
            org_z[batch_count] = static_cast<float>(c.origin[2]);
            dir_x[batch_count] = static_cast<float>(c.dir[0]);
            dir_y[batch_count] = static_cast<float>(c.dir[1]);
            dir_z[batch_count] = static_cast<float>(c.dir[2]);
            tnear[batch_count] = 0.0f;
            tfar_arr[batch_count] = c.tfar;
            valid[batch_count] = -1;  // -1 means valid ray
            batch_count++;
            i++;
        }

        // Mark remaining slots as invalid
        for (int j = batch_count; j < BATCH_SIZE; ++j) {
            valid[j] = 0;
            org_x[j] = org_y[j] = org_z[j] = 0.0f;
            dir_x[j] = dir_y[j] = dir_z[j] = 0.0f;
            tnear[j] = 0.0f;
            tfar_arr[j] = -1.0f;
        }

        // Create 16-wide ray packet
        RTCRayHit16 rayhit16;
        for (int j = 0; j < BATCH_SIZE; ++j) {
            rayhit16.ray.org_x[j] = org_x[j];
            rayhit16.ray.org_y[j] = org_y[j];
            rayhit16.ray.org_z[j] = org_z[j];
            rayhit16.ray.dir_x[j] = dir_x[j];
            rayhit16.ray.dir_y[j] = dir_y[j];
            rayhit16.ray.dir_z[j] = dir_z[j];
            rayhit16.ray.tnear[j] = tnear[j];
            rayhit16.ray.tfar[j] = tfar_arr[j];
            rayhit16.ray.mask[j] = 0xFFFFFFFF;
            rayhit16.ray.flags[j] = 0;
            rayhit16.ray.time[j] = 0.0f;
            rayhit16.ray.id[j] = static_cast<unsigned int>(batch_start + j);
            rayhit16.hit.geomID[j] = RTC_INVALID_GEOMETRY_ID;
            rayhit16.hit.primID[j] = RTC_INVALID_GEOMETRY_ID;
            rayhit16.hit.instID[0][j] = RTC_INVALID_GEOMETRY_ID;
        }

        // Execute 16-wide ray trace
        rtcIntersect16(valid, scene, &rayhit16);

        // Extract results
        for (int j = 0; j < batch_count; ++j) {
            visible[batch_start + j] = (rayhit16.hit.geomID[j] == RTC_INVALID_GEOMETRY_ID);
        }
    }
}

// AVX2 8-wide ray packet visibility test
static void batchedVisibilityTest8(
    RTCScene scene,
    const std::vector<RayCandidate>& candidates,
    std::vector<bool>& visible,
    size_t start_idx,
    size_t end_idx) {

    constexpr int BATCH_SIZE = 8;

    // 32-byte aligned arrays for AVX2
    alignas(32) float org_x[BATCH_SIZE];
    alignas(32) float org_y[BATCH_SIZE];
    alignas(32) float org_z[BATCH_SIZE];
    alignas(32) float dir_x[BATCH_SIZE];
    alignas(32) float dir_y[BATCH_SIZE];
    alignas(32) float dir_z[BATCH_SIZE];
    alignas(32) float tnear[BATCH_SIZE];
    alignas(32) float tfar_arr[BATCH_SIZE];
    alignas(32) int valid[BATCH_SIZE];

    size_t i = start_idx;
    while (i < end_idx) {
        int batch_count = 0;
        size_t batch_start = i;

        // Fill batch
        while (batch_count < BATCH_SIZE && i < end_idx) {
            const auto& c = candidates[i];
            org_x[batch_count] = static_cast<float>(c.origin[0]);
            org_y[batch_count] = static_cast<float>(c.origin[1]);
            org_z[batch_count] = static_cast<float>(c.origin[2]);
            dir_x[batch_count] = static_cast<float>(c.dir[0]);
            dir_y[batch_count] = static_cast<float>(c.dir[1]);
            dir_z[batch_count] = static_cast<float>(c.dir[2]);
            tnear[batch_count] = 0.0f;
            tfar_arr[batch_count] = c.tfar;
            valid[batch_count] = -1;  // -1 means valid ray
            batch_count++;
            i++;
        }

        // Mark remaining slots as invalid
        for (int j = batch_count; j < BATCH_SIZE; ++j) {
            valid[j] = 0;
            org_x[j] = org_y[j] = org_z[j] = 0.0f;
            dir_x[j] = dir_y[j] = dir_z[j] = 0.0f;
            tnear[j] = 0.0f;
            tfar_arr[j] = -1.0f;
        }

        // Create 8-wide ray packet
        RTCRayHit8 rayhit8;
        for (int j = 0; j < BATCH_SIZE; ++j) {
            rayhit8.ray.org_x[j] = org_x[j];
            rayhit8.ray.org_y[j] = org_y[j];
            rayhit8.ray.org_z[j] = org_z[j];
            rayhit8.ray.dir_x[j] = dir_x[j];
            rayhit8.ray.dir_y[j] = dir_y[j];
            rayhit8.ray.dir_z[j] = dir_z[j];
            rayhit8.ray.tnear[j] = tnear[j];
            rayhit8.ray.tfar[j] = tfar_arr[j];
            rayhit8.ray.mask[j] = 0xFFFFFFFF;
            rayhit8.ray.flags[j] = 0;
            rayhit8.ray.time[j] = 0.0f;
            rayhit8.ray.id[j] = static_cast<unsigned int>(batch_start + j);
            rayhit8.hit.geomID[j] = RTC_INVALID_GEOMETRY_ID;
            rayhit8.hit.primID[j] = RTC_INVALID_GEOMETRY_ID;
            rayhit8.hit.instID[0][j] = RTC_INVALID_GEOMETRY_ID;
        }

        // Execute 8-wide ray trace
        rtcIntersect8(valid, scene, &rayhit8);

        // Extract results
        for (int j = 0; j < batch_count; ++j) {
            visible[batch_start + j] = (rayhit8.hit.geomID[j] == RTC_INVALID_GEOMETRY_ID);
        }
    }
}

// Batched visibility test using Embree ray packets
// Automatically selects AVX-512 (16-wide) or AVX2 (8-wide) based on CPU support
static void batchedVisibilityTest(
    RTCScene scene,
    const std::vector<RayCandidate>& candidates,
    std::vector<bool>& visible) {

    visible.resize(candidates.size());
    if (candidates.empty()) return;

    const size_t n_candidates = candidates.size();
    static bool use_avx512 = hasAVX512();
    static bool logged = false;
    if (!logged) {
        logged = true;
        if (use_avx512) {
            std::cout << "  Using AVX-512 16-wide ray packets for visibility testing" << std::endl;
        } else {
            std::cout << "  Using AVX2 8-wide ray packets for visibility testing" << std::endl;
        }
    }

    if (use_avx512) {
        batchedVisibilityTest16(scene, candidates, visible, 0, n_candidates);
    } else {
        batchedVisibilityTest8(scene, candidates, visible, 0, n_candidates);
    }
}

void HMatrixViewFactor::computeViewFactorBlock(
    const std::vector<int>& row_faces,
    const std::vector<int>& col_faces,
    SparseBlock& sparse) {

    sparse.n_rows = static_cast<int>(row_faces.size());
    sparse.n_cols = static_cast<int>(col_faces.size());
    sparse.row_ptrs.resize(sparse.n_rows + 1, 0);

    // For small blocks, use the original single-ray approach (lower overhead)
    // For larger blocks, use batched ray queries (better throughput)
    constexpr size_t BATCH_THRESHOLD = 1000;  // Use batching for >1000 face pairs

    if (row_faces.size() * col_faces.size() < BATCH_THRESHOLD) {
        // Original single-ray approach for small blocks
        std::vector<std::vector<std::pair<int, double>>> row_entries(sparse.n_rows);

        for (size_t i = 0; i < row_faces.size(); ++i) {
            int face_i = row_faces[i];

            for (size_t j = 0; j < col_faces.size(); ++j) {
                int face_j = col_faces[j];
                if (face_i == face_j) continue;

                double F_ij = computeViewFactor(face_i, face_j);
                if (keepPairEntryF(F_ij, face_areas_[face_i], face_areas_[face_j],
                                   rad_config_.visibility_threshold)) {
                    row_entries[i].push_back({static_cast<int>(j), F_ij});
                }
            }
        }

        // Convert to CSR format
        for (int i = 0; i < sparse.n_rows; ++i) {
            sparse.row_ptrs[i + 1] = sparse.row_ptrs[i] +
                                     static_cast<int>(row_entries[i].size());
        }

        size_t nnz = sparse.row_ptrs[sparse.n_rows];
        sparse.values.resize(nnz);
        sparse.col_indices.resize(nnz);

        for (int i = 0; i < sparse.n_rows; ++i) {
            int pos = sparse.row_ptrs[i];
            std::sort(row_entries[i].begin(), row_entries[i].end());
            for (const auto& entry : row_entries[i]) {
                sparse.col_indices[pos] = entry.first;
                sparse.values[pos] = entry.second;
                pos++;
            }
        }
        return;
    }

    // PHASE 1: Parallel geometric culling - gather candidates per row
    std::vector<std::vector<RayCandidate>> row_candidates(row_faces.size());

    // Conservative reserve: expect ~100-500 visible neighbors per face, not 25% of all faces
    const size_t reserve_per_row = std::min(static_cast<size_t>(500), col_faces.size());

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < row_faces.size(); ++i) {
        int face_i = row_faces[i];
        const auto& c_i = face_centers_[face_i];
        const auto& n_i = vfNormal(face_i);

        row_candidates[i].reserve(reserve_per_row);

        for (size_t j = 0; j < col_faces.size(); ++j) {
            int face_j = col_faces[j];
            if (face_i == face_j) continue;

            const auto& c_j = face_centers_[face_j];
            const auto& n_j = vfNormal(face_j);
            double A_j = face_areas_[face_j];

            // Geometric culling
            auto r_vec = subtract(c_j, c_i);
            double r2 = dot(r_vec, r_vec);
            if (r2 < 1e-20) continue;

            double r = std::sqrt(r2);
            std::array<double, 3> r_hat = {{r_vec[0] / r, r_vec[1] / r, r_vec[2] / r}};

            // Back-facing check
            double dot_i = dot(n_i, r_hat);
            double dot_j = dot(n_j, r_hat);
            if (dot_i <= 0.0 || dot_j >= 0.0) continue;

            double cos_i = dot_i;
            double cos_j = -dot_j;

            // Grazing angle rejection, the SHARED cut (see computeViewFactorRow).
            if (cos_i < kGrazingCosCut || cos_j < kGrazingCosCut) continue;

            // Distance-based culling
            double estimated_F = (A_j / M_PI) * (cos_i * cos_j) / r2;
            // Pre-cull on the pair's strongest direction, and ONLY where M1 is
            // the answer. In the near field it is not -- it is the estimator
            // the refinement exists to replace -- and on a concave crease the
            // centroid-to-centroid line can be grazing while the sub-triangles
            // see each other perfectly well, so a near-zero M1 says nothing
            // about the refined value. Near pairs therefore bypass this gate.
            if (!keepPairEntryF(estimated_F, face_areas_[face_i], A_j,
                                rad_config_.visibility_threshold * 0.1)) {
                if (legacyPrecullRequested() ||
                    !isNearFieldPair(face_i, face_j, std::sqrt(r2))) continue;
                g_precull_near_bypassed++;
            }

            // Passed all culling - add to batch
            RayCandidate cand;
            cand.local_row = static_cast<int>(i);
            cand.local_col = static_cast<int>(j);
            cand.estimated_F = estimated_F;
            // Refine near-field pairs, where the M1 centroid estimate overshoots
            // (this is the source of Sum_j F_ij > 1).
            if (isNearFieldPair(face_i, face_j, r)) {
                cand.estimated_F = computeViewFactorRefined(face_i, face_j);
            }

            setCanonicalVisibilityRay(face_i, face_j, r, cand);


            row_candidates[i].push_back(cand);
        }
    }

    // Flatten candidates and build index map
    std::vector<RayCandidate> all_candidates;
    size_t total_candidates = 0;
    for (const auto& rc : row_candidates) {
        total_candidates += rc.size();
    }
    all_candidates.reserve(total_candidates);

    for (const auto& rc : row_candidates) {
        for (const auto& cand : rc) {
            all_candidates.push_back(cand);
        }
    }

    // PHASE 2: Batched visibility testing
    std::vector<bool> visible(all_candidates.size(), true);

    if (!all_candidates.empty()) {
        // Separate candidates needing ray trace
        std::vector<RayCandidate> ray_candidates;
        std::vector<size_t> ray_indices;
        ray_candidates.reserve(all_candidates.size());

        for (size_t k = 0; k < all_candidates.size(); ++k) {
            if (all_candidates[k].tfar >= 0.1f) {
                ray_indices.push_back(k);
                ray_candidates.push_back(all_candidates[k]);
            }
        }

        if (!ray_candidates.empty()) {
            std::vector<bool> ray_visible;
            if (use_horizon_vis_ && horizon_.built()) {
                // Heightfield-horizon occlusion (robust at grazing tangent).
                ray_visible.assign(ray_candidates.size(), true);
                for (size_t kk = 0; kk < ray_candidates.size(); ++kk)
                    ray_visible[kk] = !horizon_.occluded(
                        row_faces[ray_candidates[kk].local_row],
                        col_faces[ray_candidates[kk].local_col]);
            } else {
                batchedVisibilityTest(scene_, ray_candidates, ray_visible);
            }

            for (size_t k = 0; k < ray_candidates.size(); ++k) {
                visible[ray_indices[k]] = ray_visible[k];
            }
        }
    }

    // PHASE 3: Build sparse matrix
    std::vector<std::vector<std::pair<int, double>>> row_entries(sparse.n_rows);

    for (size_t k = 0; k < all_candidates.size(); ++k) {
        if (visible[k] && all_candidates[k].estimated_F > rad_config_.visibility_threshold) {
            row_entries[all_candidates[k].local_row].push_back({
                all_candidates[k].local_col,
                all_candidates[k].estimated_F
            });
        }
    }

    // Convert to CSR format
    for (int i = 0; i < sparse.n_rows; ++i) {
        sparse.row_ptrs[i + 1] = sparse.row_ptrs[i] +
                                 static_cast<int>(row_entries[i].size());
    }

    size_t nnz = sparse.row_ptrs[sparse.n_rows];
    sparse.values.resize(nnz);
    sparse.col_indices.resize(nnz);

    for (int i = 0; i < sparse.n_rows; ++i) {
        int pos = sparse.row_ptrs[i];
        std::sort(row_entries[i].begin(), row_entries[i].end());
        for (const auto& entry : row_entries[i]) {
            sparse.col_indices[pos] = entry.first;
            sparse.values[pos] = entry.second;
            pos++;
        }
    }
}

// ============================================================================
// Randomized SVD Implementation
// ============================================================================
// Based on Halko, Martinsson, and Tropp (2011) "Finding structure with randomness"
// Algorithm: Randomized range finder + standard SVD on smaller matrix
//
// For an m x n matrix A with target rank k:
// 1. Draw random matrix Omega (n x (k+p)), where p is oversampling
// 2. Form Y = A * Omega (m x (k+p))
// 3. Power iteration: Y = A * (A^T * Y) repeated q times
// 4. Compute QR: Y = Q * R, where Q is m x (k+p)
// 5. Form B = Q^T * A (small matrix: (k+p) x n)
// 6. Compute SVD of B: B = U_B * Sigma * V^T
// 7. Final: U = Q * U_B
//
// Complexity: O(m*n*k) instead of O(m*n*min(m,n))

struct RandomizedSVDResult {
    Eigen::MatrixXd U;      // Left singular vectors (m x rank)
    Eigen::VectorXd sigma;  // Singular values (rank)
    Eigen::MatrixXd V;      // Right singular vectors (n x rank)
    int rank = 0;
    bool success = false;
};

static RandomizedSVDResult computeRandomizedSVD(
    const Eigen::MatrixXd& A,
    int target_rank,
    int oversampling = 10,
    int power_iterations = 2) {

    RandomizedSVDResult result;

    int m = static_cast<int>(A.rows());
    int n = static_cast<int>(A.cols());

    if (m == 0 || n == 0) {
        return result;
    }

    // Effective rank (with oversampling, but capped at matrix dimensions)
    int k = std::min(target_rank + oversampling, std::min(m, n));

    // For very small matrices, fall back to exact SVD
    if (k >= std::min(m, n) - 1) {
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        result.U = svd.matrixU();
        result.sigma = svd.singularValues();
        result.V = svd.matrixV();
        result.rank = static_cast<int>(result.sigma.size());
        result.success = true;
        return result;
    }

    // Step 1: Draw random Gaussian matrix
    static thread_local std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::normal_distribution<double> dist(0.0, 1.0);

    Eigen::MatrixXd Omega(n, k);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            Omega(i, j) = dist(gen);
        }
    }

    // Step 2: Form Y = A * Omega
    Eigen::MatrixXd Y = A * Omega;

    // Step 3: Power iterations for improved accuracy
    // Y = A * (A^T * Y), which emphasizes the dominant singular vectors
    for (int iter = 0; iter < power_iterations; ++iter) {
        // Y = A * A^T * Y
        Eigen::MatrixXd Z = A.transpose() * Y;
        Y = A * Z;
    }

    // Step 4: QR decomposition to get orthonormal basis Q for range of A
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(Y);
    Eigen::MatrixXd Q = qr.householderQ() * Eigen::MatrixXd::Identity(m, k);

    // Step 5: Form the small matrix B = Q^T * A
    Eigen::MatrixXd B = Q.transpose() * A;

    // Step 6: Compute SVD of the small matrix B
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(B, Eigen::ComputeThinU | Eigen::ComputeThinV);

    // Step 7: Recover left singular vectors: U = Q * U_B
    result.U = Q * svd.matrixU();
    result.sigma = svd.singularValues();
    result.V = svd.matrixV();
    result.rank = static_cast<int>(result.sigma.size());
    result.success = true;

    return result;
}

bool HMatrixViewFactor::tryCompressToLowRank(
    const SparseBlock& sparse,
    LowRankBlock& lowrank,
    double tolerance) {

    if (sparse.values.empty()) return false;

    int m = sparse.n_rows;
    int n = sparse.n_cols;

    // Convert sparse to dense Eigen matrix for SVD
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(m, n);
    for (int i = 0; i < m; ++i) {
        for (int k = sparse.row_ptrs[i]; k < sparse.row_ptrs[i + 1]; ++k) {
            A(i, sparse.col_indices[k]) = sparse.values[k];
        }
    }

    // Choose SVD method based on configuration
    Eigen::MatrixXd U_full;
    Eigen::VectorXd sigma;
    Eigen::MatrixXd V_full;

    if (config_.svd_method == SVDMethod::Randomized) {
        // Use randomized SVD for large matrices
        // Target rank is max_svd_rank, but actual rank will be determined by tolerance
        auto rsvd = computeRandomizedSVD(
            A,
            config_.max_svd_rank,
            config_.randomized_oversampling,
            config_.randomized_power_iterations);

        if (!rsvd.success || rsvd.sigma.size() == 0) return false;

        U_full = std::move(rsvd.U);
        sigma = std::move(rsvd.sigma);
        V_full = std::move(rsvd.V);
    } else {
        // Use exact JacobiSVD
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        U_full = svd.matrixU();
        sigma = svd.singularValues();
        V_full = svd.matrixV();
    }

    if (sigma.size() == 0 || sigma(0) < 1e-15) return false;

    // Find truncation rank based on tolerance
    int rank = 0;
    double threshold = tolerance * sigma(0);
    for (int i = 0; i < sigma.size(); ++i) {
        if (sigma(i) > threshold) {
            rank = i + 1;
        }
    }

    if (rank == 0) return false;

    // Cap rank. When the cap binds, the stored block does NOT meet `tolerance`;
    // record by how much instead of returning a quiet success. sigma(rank_cap)
    // / sigma(0) is the relative size of the largest singular value thrown
    // away, which is the tolerance actually delivered.
    g_svd_compress_calls++;
    const int rank_wanted = rank;
    rank = std::min(rank, config_.max_svd_rank);
    rank = std::min(rank, static_cast<int>(sigma.size()));
    if (rank_wanted > rank) {
        g_svd_rank_capped++;
        const double resid = (sigma(0) > 0.0) ? sigma(rank) / sigma(0) : 0.0;
        double prev = g_svd_rank_cap_worst_resid.load(std::memory_order_relaxed);
        while (resid > prev &&
               !g_svd_rank_cap_worst_resid.compare_exchange_weak(prev, resid)) {}
        int prev_want = g_svd_rank_cap_worst_want.load(std::memory_order_relaxed);
        while (rank_wanted > prev_want &&
               !g_svd_rank_cap_worst_want.compare_exchange_weak(prev_want, rank_wanted)) {}
    }

    // Check if low-rank is beneficial
    size_t sparse_bytes = sparse.memoryBytes();
    size_t lowrank_bytes = (m + n) * rank * sizeof(double) + rank * sizeof(double);

    if (lowrank_bytes >= sparse_bytes * 0.9) {
        // Not worth compressing
        return false;
    }

    // Build low-rank approximation
    lowrank.n_rows = m;
    lowrank.n_cols = n;
    lowrank.rank = rank;
    lowrank.U = U_full.leftCols(rank);
    lowrank.sigma = sigma.head(rank);
    lowrank.V = V_full.leftCols(rank);

    return true;
}

// =========== ACA (Adaptive Cross Approximation) Implementation ===========

void HMatrixViewFactor::computeViewFactorRow(
    int local_row,
    const std::vector<int>& row_faces,
    const std::vector<int>& col_faces,
    std::vector<double>& result) const {

    result.assign(col_faces.size(), 0.0);

    int face_i = row_faces[local_row];
    const auto& c_i = face_centers_[face_i];
    const auto& n_i = vfNormal(face_i);

    // Collect candidates that pass geometric culling
    std::vector<RayCandidate> candidates;
    candidates.reserve(std::min(size_t(500), col_faces.size()));

    for (size_t j = 0; j < col_faces.size(); ++j) {
        int face_j = col_faces[j];
        if (face_i == face_j) continue;

        const auto& c_j = face_centers_[face_j];
        const auto& n_j = vfNormal(face_j);
        double A_j = face_areas_[face_j];

        // Geometric culling (same as computeViewFactorBlock)
        auto r_vec = subtract(c_j, c_i);
        double r2 = dot(r_vec, r_vec);
        if (r2 < 1e-20) continue;

        double r = std::sqrt(r2);
        std::array<double, 3> r_hat = {{r_vec[0] / r, r_vec[1] / r, r_vec[2] / r}};

        // Back-facing check
        double dot_i = dot(n_i, r_hat);
        double dot_j = dot(n_j, r_hat);
        if (dot_i <= 0.0 || dot_j >= 0.0) continue;

        double cos_i = dot_i;
        double cos_j = -dot_j;

        // Grazing angle rejection, the SHARED cut. It was 0.01 here and 1e-6
        // on the sparse builder, so this builder deleted pairs the other one
        // kept -- a disagreement about which entries exist, upstream of any
        // disagreement about their values.
        if (cos_i < kGrazingCosCut || cos_j < kGrazingCosCut) continue;

        // Distance-based culling
        double estimated_F = (A_j / M_PI) * (cos_i * cos_j) / r2;
        // See computeViewFactorBlock: near pairs bypass the M1 pre-cull.
        if (!keepPairEntryF(estimated_F, face_areas_[face_i], A_j,
                            rad_config_.visibility_threshold * 0.1)) {
            if (legacyPrecullRequested() ||
                !isNearFieldPair(face_i, face_j, std::sqrt(r2))) continue;
            g_precull_near_bypassed++;
        }

        // Passed all culling - add to batch
        RayCandidate cand;
        cand.local_row = local_row;
        cand.local_col = static_cast<int>(j);
        cand.estimated_F = estimated_F;
        noteCrossPair(r, rad_config_.nearfield_window_factor *
                            (std::sqrt(face_areas_[face_i]) + std::sqrt(A_j)));
        // Same near-field refinement the dense-block generator applies. Without
        // it, whether a close pair got the M2 quadrature depended on which
        // BLOCK the octree put it in -- ACA samples the matrix through THIS
        // function, so a low-rank block was built from unrefined M1 entries.
        if (isNearFieldPair(face_i, face_j, r)) {
            cand.estimated_F = computeViewFactorRefined(face_i, face_j);
        }

        setCanonicalVisibilityRay(face_i, face_j, r, cand);


        candidates.push_back(cand);
    }

    if (candidates.empty()) return;

    // Batched visibility testing
    std::vector<bool> visible(candidates.size(), true);

    // Separate candidates needing ray trace
    std::vector<RayCandidate> ray_candidates;
    std::vector<size_t> ray_indices;
    ray_candidates.reserve(candidates.size());

    for (size_t k = 0; k < candidates.size(); ++k) {
        if (candidates[k].tfar >= 0.1f) {
            ray_indices.push_back(k);
            ray_candidates.push_back(candidates[k]);
        }
    }

    if (!ray_candidates.empty()) {
        std::vector<bool> ray_visible;
        batchedVisibilityTest(scene_, ray_candidates, ray_visible);

        for (size_t k = 0; k < ray_candidates.size(); ++k) {
            visible[ray_indices[k]] = ray_visible[k];
        }
    }

    // Write results
    for (size_t k = 0; k < candidates.size(); ++k) {
        if (visible[k] && candidates[k].estimated_F > rad_config_.visibility_threshold) {
            result[candidates[k].local_col] = candidates[k].estimated_F;
        }
    }
}

void HMatrixViewFactor::computeViewFactorColumn(
    int local_col,
    const std::vector<int>& row_faces,
    const std::vector<int>& col_faces,
    std::vector<double>& result) const {

    result.assign(row_faces.size(), 0.0);

    int face_j = col_faces[local_col];
    const auto& c_j = face_centers_[face_j];
    const auto& n_j = vfNormal(face_j);
    double A_j = face_areas_[face_j];

    // Collect candidates that pass geometric culling
    std::vector<RayCandidate> candidates;
    candidates.reserve(std::min(size_t(500), row_faces.size()));

    for (size_t i = 0; i < row_faces.size(); ++i) {
        int face_i = row_faces[i];
        if (face_i == face_j) continue;

        const auto& c_i = face_centers_[face_i];
        const auto& n_i = vfNormal(face_i);

        // Geometric culling (same as computeViewFactorBlock)
        auto r_vec = subtract(c_j, c_i);
        double r2 = dot(r_vec, r_vec);
        if (r2 < 1e-20) continue;

        double r = std::sqrt(r2);
        std::array<double, 3> r_hat = {{r_vec[0] / r, r_vec[1] / r, r_vec[2] / r}};

        // Back-facing check
        double dot_i = dot(n_i, r_hat);
        double dot_j = dot(n_j, r_hat);
        if (dot_i <= 0.0 || dot_j >= 0.0) continue;

        double cos_i = dot_i;
        double cos_j = -dot_j;

        // Grazing angle rejection, the SHARED cut. It was 0.01 here and 1e-6
        // on the sparse builder, so this builder deleted pairs the other one
        // kept -- a disagreement about which entries exist, upstream of any
        // disagreement about their values.
        if (cos_i < kGrazingCosCut || cos_j < kGrazingCosCut) continue;

        // Distance-based culling
        double estimated_F = (A_j / M_PI) * (cos_i * cos_j) / r2;
        // See computeViewFactorBlock: near pairs bypass the M1 pre-cull.
        if (!keepPairEntryF(estimated_F, face_areas_[face_i], A_j,
                            rad_config_.visibility_threshold * 0.1)) {
            if (legacyPrecullRequested() ||
                !isNearFieldPair(face_i, face_j, std::sqrt(r2))) continue;
            g_precull_near_bypassed++;
        }

        // Passed all culling - add to batch
        RayCandidate cand;
        cand.local_row = static_cast<int>(i);
        cand.local_col = local_col;
        cand.estimated_F = estimated_F;
        noteCrossPair(r, rad_config_.nearfield_window_factor *
                            (std::sqrt(face_areas_[face_i]) + std::sqrt(A_j)));
        // See computeViewFactorRow: the same near-field refinement, for the same
        // reason. F_ij is the ROW face's view factor either way, so the call
        // takes (face_i, face_j) in that order here too.
        if (isNearFieldPair(face_i, face_j, r)) {
            cand.estimated_F = computeViewFactorRefined(face_i, face_j);
        }

        // Ray goes from face_i to face_j
        setCanonicalVisibilityRay(face_i, face_j, r, cand);


        candidates.push_back(cand);
    }

    if (candidates.empty()) return;

    // Batched visibility testing
    std::vector<bool> visible(candidates.size(), true);

    // Separate candidates needing ray trace
    std::vector<RayCandidate> ray_candidates;
    std::vector<size_t> ray_indices;
    ray_candidates.reserve(candidates.size());

    for (size_t k = 0; k < candidates.size(); ++k) {
        if (candidates[k].tfar >= 0.1f) {
            ray_indices.push_back(k);
            ray_candidates.push_back(candidates[k]);
        }
    }

    if (!ray_candidates.empty()) {
        std::vector<bool> ray_visible;
        batchedVisibilityTest(scene_, ray_candidates, ray_visible);

        for (size_t k = 0; k < ray_candidates.size(); ++k) {
            visible[ray_indices[k]] = ray_visible[k];
        }
    }

    // Write results
    for (size_t k = 0; k < candidates.size(); ++k) {
        if (visible[k] && candidates[k].estimated_F > rad_config_.visibility_threshold) {
            result[candidates[k].local_row] = candidates[k].estimated_F;
        }
    }
}

bool HMatrixViewFactor::tryCompressToLowRankACA(
    const std::vector<int>& row_faces,
    const std::vector<int>& col_faces,
    LowRankBlock& lowrank,
    double tolerance) {

    g_aca_attempts++;

    int m = static_cast<int>(row_faces.size());
    int n = static_cast<int>(col_faces.size());

    // Skip ACA for very small blocks
    if (m < config_.aca_min_block_size || n < config_.aca_min_block_size) {
        g_aca_fail_small++;
        return false;  // Fall back to SVD
    }

    int max_rank = std::min({config_.aca_max_rank, m, n});

    // Storage for U and V columns (grow incrementally)
    std::vector<Eigen::VectorXd> U_cols;
    std::vector<Eigen::VectorXd> V_cols;
    U_cols.reserve(max_rank);
    V_cols.reserve(max_rank);

    // Track used pivots
    std::vector<bool> used_rows(m, false);
    std::vector<bool> used_cols(n, false);

    // Estimate of ||F|| for relative tolerance
    double F_norm_est = 0.0;

    // First pivot row (start with row 0)
    int i_pivot = 0;

    // Working vectors
    std::vector<double> row_vec(n);
    std::vector<double> col_vec(m);

    for (int k = 0; k < max_rank; ++k) {
        // Compute row i_pivot of the view factor matrix
        computeViewFactorRow(i_pivot, row_faces, col_faces, row_vec);

        // Subtract current approximation: row_vec -= U[i_pivot,:] * V^T
        for (size_t r = 0; r < U_cols.size(); ++r) {
            double u_ir = U_cols[r](i_pivot);
            for (int j = 0; j < n; ++j) {
                row_vec[j] -= u_ir * V_cols[r](j);
            }
        }

        // Find pivot column (max absolute value in residual row, excluding used columns)
        int j_pivot = -1;
        double max_val = 0.0;
        for (int j = 0; j < n; ++j) {
            if (!used_cols[j] && std::abs(row_vec[j]) > max_val) {
                max_val = std::abs(row_vec[j]);
                j_pivot = j;
            }
        }

        if (j_pivot < 0) {
            // Robust ACA pivoting. The current row's residual is zero on all
            // unused columns (e.g. a floor face that is fully occluded from this
            // wall cluster, common with visibility-structured view-factor blocks).
            // A naive ACA stops here and returns rank 0, after which the caller
            // ZEROS the entire large admissible block -- discarding the
            // floor<->wall IR coupling for every other face in it (the PSR-floor
            // decoupling seen at >100k faces). Instead, skip to the next unused
            // row and retry without consuming a rank term. Bounded by m skips.
            used_rows[i_pivot] = true;
            int alt = -1;
            for (int i = 0; i < m; ++i) { if (!used_rows[i]) { alt = i; break; } }
            if (alt < 0) break;   // every row exhausted -> genuinely empty block
            i_pivot = alt;
            --k;                  // this sweep produced no rank-1 update
            continue;
        }

        double gamma = row_vec[j_pivot];

        // Update norm estimate on first iteration
        if (k == 0) {
            F_norm_est = max_val;
        }

        // Check convergence: if pivot element is small relative to norm estimate
        if (std::abs(gamma) < tolerance * F_norm_est || std::abs(gamma) < 1e-15) {
            break;  // Converged
        }

        // Compute column j_pivot of the view factor matrix
        computeViewFactorColumn(j_pivot, row_faces, col_faces, col_vec);

        // Subtract current approximation: col_vec -= U * V[j_pivot,:]
        for (size_t r = 0; r < U_cols.size(); ++r) {
            double v_jr = V_cols[r](j_pivot);
            for (int i = 0; i < m; ++i) {
                col_vec[i] -= U_cols[r](i) * v_jr;
            }
        }

        // Create new rank-1 update vectors
        Eigen::VectorXd u_new(m), v_new(n);
        for (int i = 0; i < m; ++i) u_new(i) = col_vec[i] / gamma;
        for (int j = 0; j < n; ++j) v_new(j) = row_vec[j];

        U_cols.push_back(u_new);
        V_cols.push_back(v_new);

        // Update norm estimate
        double update_norm = u_new.norm() * v_new.norm();
        F_norm_est = std::max(F_norm_est, update_norm);

        // Mark pivots as used
        used_rows[i_pivot] = true;
        used_cols[j_pivot] = true;

        // Select next row pivot (max entry in new column vector among unused rows)
        double max_u = 0.0;
        int next_pivot = -1;
        for (int i = 0; i < m; ++i) {
            if (!used_rows[i] && std::abs(u_new(i)) > max_u) {
                max_u = std::abs(u_new(i));
                next_pivot = i;
            }
        }

        if (next_pivot < 0 || max_u < 1e-15) break;  // No more significant rows
        i_pivot = next_pivot;
    }

    int rank = static_cast<int>(U_cols.size());
    if (rank == 0) {
        g_aca_fail_norank++;
        return false;
    }

    // Check if ACA provides compression benefit vs dense storage
    // NOTE: We now ALWAYS accept the ACA result, even if rank is high.
    // The fallback to full sparse computation uses O(n²) memory which can cause OOM
    // with many threads. It's better to store a higher-rank approximation than to
    // risk falling back to the memory-expensive full computation.
    //
    // For very high rank (>90% of min(m,n)), we could consider hierarchical subdivision,
    // but that's handled by the caller based on the return value.
    size_t aca_bytes = (m + n) * rank * sizeof(double) + rank * sizeof(double);
    size_t dense_bytes = static_cast<size_t>(m) * n * sizeof(double);
    if (aca_bytes >= dense_bytes && rank >= config_.aca_max_rank) {
        // ACA reached max rank and still doesn't compress well.
        // Return false to signal caller should try subdivision instead.
        // But ONLY if we hit max_rank - otherwise just keep the result.
        g_aca_fail_maxrank++;
        return false;
    }

    // Assemble into LowRankBlock
    // Store as U * V^T directly (sigma = ones vector)
    lowrank.n_rows = m;
    lowrank.n_cols = n;
    lowrank.rank = rank;
    lowrank.U.resize(m, rank);
    lowrank.V.resize(n, rank);
    lowrank.sigma = Eigen::VectorXd::Ones(rank);

    for (int r = 0; r < rank; ++r) {
        lowrank.U.col(r) = U_cols[r];
        lowrank.V.col(r) = V_cols[r];
    }

    g_aca_successes++;
    return true;
}

// =========== End ACA Implementation ===========

bool HMatrixViewFactor::isAdmissible(QuadtreeNode* row_node, QuadtreeNode* col_node) const {
    // Standard admissibility condition:
    // max(diam(X), diam(Y)) <= eta * dist(X, Y)
    //
    // We use eta = 0.8 (a common choice)

    const double eta = 0.8;

    // Distance between cluster centers (2D)
    double dx = row_node->center[0] - col_node->center[0];
    double dy = row_node->center[1] - col_node->center[1];
    double dist = std::sqrt(dx * dx + dy * dy);

    double max_diam = std::max(row_node->diameter, col_node->diameter);

    return max_diam <= eta * dist;
}

// Octree version - uses 3D distance for better admissibility in crater geometry
bool HMatrixViewFactor::isAdmissible(OctreeNode* row_node, OctreeNode* col_node) const {
    // Standard admissibility condition:
    // max(diam(X), diam(Y)) <= eta * dist(X, Y)
    //
    // We use eta = 0.8 (a common choice)
    // The 3D distance is more accurate for crater geometry where Z matters

    const double eta = 0.8;

    // Distance between cluster centers (3D - includes Z component)
    double dx = row_node->center[0] - col_node->center[0];
    double dy = row_node->center[1] - col_node->center[1];
    double dz = row_node->center[2] - col_node->center[2];
    double dist = std::sqrt(dx * dx + dy * dy + dz * dz);

    double max_diam = std::max(row_node->diameter, col_node->diameter);

    return max_diam <= eta * dist;
}

double HMatrixViewFactor::computeViewFactor(int face_i, int face_j) const {
    // Use centroid approximation (M1 method from Rezac & Zhao 2020)
    // F_ij = (A_j / π) * (cos_i * cos_j) / r²

    const auto& c_i = face_centers_[face_i];
    const auto& c_j = face_centers_[face_j];
    const auto& n_i = vfNormal(face_i);
    const auto& n_j = vfNormal(face_j);
    double A_j = face_areas_[face_j];

    auto r_vec = subtract(c_j, c_i);
    double r2 = dot(r_vec, r_vec);
    if (r2 < 1e-20) return 0.0;

    double r = std::sqrt(r2);
    std::array<double, 3> r_hat = {{r_vec[0] / r, r_vec[1] / r, r_vec[2] / r}};

    // SPATIAL CULLING OPTIMIZATION 1: Back-facing geometry check
    // For face i to see face j:
    //   - n_i must point toward j (dot(n_i, r_hat) > 0)
    //   - n_j must point toward i (dot(n_j, -r_hat) > 0, i.e., dot(n_j, r_hat) < 0)
    // This is a cheap geometric test that can eliminate many pairs without ray tracing.
    double dot_i = dot(n_i, r_hat);  // positive if n_i points toward j
    double dot_j = dot(n_j, r_hat);  // negative if n_j points toward i

    // For mutual visibility, we need dot_i > 0 AND dot_j < 0
    // If both normals point the same way relative to r_hat, faces are back-to-back
    if (dot_i <= 0.0 || dot_j >= 0.0) {
        return 0.0;  // Back-facing or parallel - no visibility possible
    }

    double cos_i = dot_i;     // Already positive
    double cos_j = -dot_j;    // Convert to positive (face j looking toward i)

    // SPATIAL CULLING OPTIMIZATION 2: Grazing angle rejection, the SHARED cut
    // (see computeViewFactorRow for why it is no longer 0.01 here).
    if (cos_i < kGrazingCosCut || cos_j < kGrazingCosCut) {
        return 0.0;
    }

    // SPATIAL CULLING OPTIMIZATION 3: Distance-based contribution estimate
    // Skip expensive ray trace if contribution would be negligible anyway
    // F_ij ~ A_j * cos_i * cos_j / (π * r²)
    // If this is below threshold, skip
    double estimated_F = (A_j / M_PI) * (cos_i * cos_j) / r2;
    // See computeViewFactorBlock: near pairs bypass the M1 pre-cull.
    if (!keepPairEntryF(estimated_F, face_areas_[face_i], A_j,
                        rad_config_.visibility_threshold * 0.1)) {
        if (legacyPrecullRequested() || !isNearFieldPair(face_i, face_j, r)) {
            return 0.0;  // M1 IS the estimator out here, and it is below the floor
        }
        g_precull_near_bypassed++;
    }

    // Passed all culling tests - now do the expensive visibility check, in
    // CANONICAL (min index first) order. Mutual visibility is symmetric;
    // isVisible() is not exactly symmetric in its arguments, and letting the
    // float32 traversal decide puts a direction-dependent zero into the matrix.
    if (!isVisible(std::min(face_i, face_j), std::max(face_i, face_j))) return 0.0;

    // M1 (centroid) estimate is a far-field approximation. In the near field
    // (centers closer than a few face-widths) it overshoots, so refine with
    // area-to-area quadrature.
    if (isNearFieldPair(face_i, face_j, r)) {
        return computeViewFactorRefined(face_i, face_j);
    }
    return estimated_F;
}

// Fill a candidate's occlusion ray for the pair (face_i, face_j), issued in
// CANONICAL order: always from the LOWER face index to the higher one,
// whichever way the caller asked.
//
// Mutual visibility is a symmetric geometric relation. isVisible() and this ray
// setup are not exactly symmetric in their arguments -- the origin is offset
// from p1 and Embree's float32 traversal can answer differently from the two
// ends of the same segment -- so a non-canonical query lets a rounding
// difference decide whether F_ij exists while F_ji does not. The offsets
// themselves are unchanged (0.1 percent of the distance or 0.5 m, whichever is
// larger, capped at 10 percent of the distance; a segment shorter than 0.1 m
// after both margins is assumed visible), and are the same arithmetic the
// sparse builder uses.
//
// The optional normal lift (PTM_HMATRIX_RAY_LIFT, default 0 and therefore
// inert) lifts along the ORIGIN face's normal, which under canonical ordering
// is the lower-indexed face rather than the asking one.
void HMatrixViewFactor::setCanonicalVisibilityRay(int face_i, int face_j, double /*r*/,
                                                  RayCandidate& cand) const {
    const int fa = std::min(face_i, face_j);
    const int fb = std::max(face_i, face_j);

    // face_centers_ are already in SCENE coordinates here (build() subtracts
    // the bbox-center shift when it fills them), so the shared kernel is handed
    // a zero shift.
    const VisibilitySegment seg = visibilityRay(face_centers_[fa], face_centers_[fb],
                                                face_areas_[fa], face_areas_[fb],
                                                {{0.0, 0.0, 0.0}});
    if (seg.unresolvable) noteVisibilityUnresolvable();

    cand.origin = seg.origin;
    cand.dir = seg.dir;
    const double rl = getRayLiftFactor();
    if (rl > 0.0) {
        const double nl = rl * std::sqrt(2.0 * face_areas_[fa]);
        cand.origin[0] += nl * face_normals_[fa][0];
        cand.origin[1] += nl * face_normals_[fa][1];
        cand.origin[2] += nl * face_normals_[fa][2];
    }
    cand.tfar = (seg.tfar < 0.0) ? -1.0f : static_cast<float>(seg.tfar);
}

bool HMatrixViewFactor::isNearFieldPair(int fi, int fj, double r) const {
    // The near/far switch, in one place for this builder and SHARED with the
    // sparse one (radiation::isNearFieldPair, nearfield_quadrature.h). It used
    // to be written out four times here -- in computeViewFactor, in
    // computeViewFactorBlock, and (silently, by omission) NOT in
    // computeViewFactorRow/Column, which is how the ACA cross generators came
    // to skip the near field entirely.
    return radiation::isNearFieldPair(r, face_areas_[fi], face_areas_[fj],
                                      rad_config_.nearfield_window_factor);
}

int HMatrixViewFactor::refinedQuadratureOrder(int fi, int fj) const {
    // The H-matrix raises the quadrature order on pairs that are close relative
    // to their own size -- m ~ 2 for adjacent pairs, growing as the pair closes
    // -- and the configured order is the FLOOR under that, not a replacement
    // for it. Measured on lunar_crater_80N (60 faces, 10,319 near pairs, vs a
    // Richardson m = 8/16 double-area reference, 2026-08-23):
    //
    //   rule                            massW    worst pair   evals/pair
    //   adaptive only (pre-2026-08-23) +0.734%       -8.33%          142
    //   fixed m = 3 (sparse path's)    +2.899%      +31.75%           81
    //   max(3, adaptive)  <- SHIPPED   +0.729%       +7.10%          191
    //
    // so pinning this path to the sparse path's fixed order -- the tidy "one
    // key, one number" unification -- would have been a 4x mass-weighted
    // accuracy REGRESSION on this builder, and is not what shipped. The key
    // governs both paths as a floor: raise it and both get finer, and the
    // H-matrix is never coarser than the sparse builder on the same pair.
    //
    // Both inputs are symmetric in (fi, fj), so the order is too.
    const double d = std::sqrt(face_areas_[fi]) + std::sqrt(face_areas_[fj]);
    auto rv = subtract(face_centers_[fj], face_centers_[fi]);
    const double r0 = std::sqrt(dot(rv, rv));
    // The same length scale as the near/far window, so the order is 1 exactly
    // at the window edge and rises as the pair closes. Tracking the configured
    // window keeps "widen the window" from silently meaning "refine the newly
    // included pairs at a different resolution than the ones already inside".
    const double w = rad_config_.nearfield_window_factor;
    int m = (r0 > 1e-12) ? static_cast<int>(std::ceil(w * d / r0))
                         : radiation::kMaxNearFieldQuadOrder;
    m = std::max(m, rad_config_.nearfield_quadrature_order);
    if (m < 1) m = 1;
    // Resolution ceiling: numerical, and also the capacity of the shared
    // kernel's on-stack sample buffers.
    if (m > radiation::kMaxNearFieldQuadOrder) m = radiation::kMaxNearFieldQuadOrder;
    return m;
}

double HMatrixViewFactor::computeViewFactorRefined(int fi, int fj) const {
    // Near-field (M2) refinement of the M1 centroid estimate. The estimator is
    // radiation::nearFieldPairIntegral() in nearfield_quadrature.h -- the SAME
    // function the sparse builder calls, so the two paths can no longer drift
    // apart. What this builder still chooses for itself is the order (above)
    // and the occlusion ray (the callers batch theirs through Embree).
    //
    // What the shared kernel adds here, which the local copy did not have:
    //
    //  * the physical bound A_i F_ij <= min(A_i, A_j), with a counted, printed
    //    tripwire. The local copy was UNBOUNDED, and this quadrature diverges
    //    on a closing concave crease: measured on two unit triangles sharing an
    //    edge, this builder's own adaptive order read F_ij = 4.59 at a 10
    //    degree dihedral and 16.75 at 5 degrees, against references 0.84 and
    //    0.93. Fabricated energy, which the downstream row-sum renormalization
    //    would have absorbed silently by zeroing the face's sky view factor.
    //  * bit-exact reciprocity. The pair is passed in canonical (min, max)
    //    index order, so A_i F_ij and A_j F_ji come from the same summation in
    //    the same sequence. The local copy summed in row-major order over face
    //    i's samples, which is not the same sequence when the pair is asked the
    //    other way round.
    //  * the 1e-6 grazing cut, matching M1 and the sparse builder. The local
    //    copy cut at exactly 0.
    const int fa = std::min(fi, fj);
    const int fb = std::max(fi, fj);
    const auto& tri_a = mesh_->faces[fa];
    const auto& tri_b = mesh_->faces[fb];

    const double g = radiation::nearFieldPairIntegral(
        mesh_->vertices[tri_a[0]], mesh_->vertices[tri_a[1]], mesh_->vertices[tri_a[2]],
        mesh_->vertices[tri_b[0]], mesh_->vertices[tri_b[1]], mesh_->vertices[tri_b[2]],
        vfNormal(fa), vfNormal(fb),
        face_areas_[fa], face_areas_[fb], refinedQuadratureOrder(fi, fj));

    return g / face_areas_[fi];
}

bool HMatrixViewFactor::isVisible(int face_a, int face_b) const {
    // ONE rule, shared with the sparse builder: radiation::visibilityRay().
    // face_centers_ are already in scene coordinates (see build()).
    const VisibilitySegment seg = visibilityRay(face_centers_[face_a], face_centers_[face_b],
                                                face_areas_[face_a], face_areas_[face_b],
                                                {{0.0, 0.0, 0.0}});
    if (seg.unresolvable) noteVisibilityUnresolvable();
    if (seg.auto_visible || seg.tfar <= 0.0) return true;

    RTCRayHit rayhit;
    rayhit.ray.org_x = static_cast<float>(seg.origin[0]);
    rayhit.ray.org_y = static_cast<float>(seg.origin[1]);
    rayhit.ray.org_z = static_cast<float>(seg.origin[2]);
    rayhit.ray.dir_x = static_cast<float>(seg.dir[0]);
    rayhit.ray.dir_y = static_cast<float>(seg.dir[1]);
    rayhit.ray.dir_z = static_cast<float>(seg.dir[2]);
    rayhit.ray.tnear = 0.0f;
    rayhit.ray.tfar = static_cast<float>(seg.tfar);
    rayhit.ray.mask = 0xFFFFFFFF;
    rayhit.ray.flags = 0;
    rayhit.ray.time = 0.0f;
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rayhit.hit.primID = RTC_INVALID_GEOMETRY_ID;
    rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

    rtcIntersect1(scene_, &rayhit);

    return rayhit.hit.geomID == RTC_INVALID_GEOMETRY_ID;
}

namespace {

// Every global row a root block writes, following hierarchical children.
// Returns false if a descendant is disk-backed, whose row map lives in a file
// and is not knowable here.
bool collectWrittenRows(const radiation::HMatrixBlock& blk, std::vector<int>& out) {
    if (const auto* hb = std::get_if<radiation::HierarchicalBlock>(&blk.data)) {
        bool ok = true;
        for (const auto& child : hb->children) {
            if (child) ok = collectWrittenRows(*child, out) && ok;
        }
        return ok;
    }
    if (std::holds_alternative<radiation::DiskBlock>(blk.data)) return false;
    out.insert(out.end(), blk.row_indices.begin(), blk.row_indices.end());
    return true;
}

} // namespace

void HMatrixViewFactor::prepareMatvecSchedule() const {
    if (matvec_schedule_ != MatvecSchedule::Unknown) return;

    const size_t nb = root_blocks_.size();
    if (nb == 0 || n_rows_ == 0) {
        matvec_schedule_ = MatvecSchedule::PerBlock;
        return;
    }

    // The negative control for the determinism fix, kept in the shipped binary
    // so the control is runnable and not a claim about a binary that no longer
    // exists. Restores the pre-2026-08-23 atomic schedule, whose results move
    // in the last bits between two runs of the same binary.
    if (const char* atomic_env = std::getenv("PTM_HMATRIX_MATVEC_ATOMIC")) {
        if (std::string(atomic_env) == "1") {
            std::cout << "[hmatrix] WARNING: PTM_HMATRIX_MATVEC_ATOMIC=1 -- matvec uses "
                         "the legacy atomic schedule. Results are NOT reproducible run "
                         "to run. This is a control, not a production setting."
                      << std::endl;
            matvec_schedule_ = MatvecSchedule::Legacy;
            return;
        }
    }

    // Candidate grouping. The octree/quadtree builder lays root blocks out as a
    // dense N x N grid, block_idx = i * N + j, and every block in grid row i
    // writes exactly the rows of tree node i -- so grid rows are the natural
    // groups. Anything else: try one group per block.
    const size_t N = static_cast<size_t>(std::llround(std::sqrt(static_cast<double>(nb))));
    const bool grid = (N * N == nb);

    std::vector<int> owner(n_rows_, -1);
    std::vector<int> rows;
    bool disjoint = true;
    bool complete = true;

    for (size_t b = 0; b < nb && disjoint && complete; ++b) {
        if (!root_blocks_[b]) continue;
        rows.clear();
        complete = collectWrittenRows(*root_blocks_[b], rows);
        if (!complete) break;
        const int g = grid ? static_cast<int>(b / N) : static_cast<int>(b);
        for (int r : rows) {
            if (r < 0 || static_cast<size_t>(r) >= n_rows_) continue;
            if (owner[r] < 0) owner[r] = g;
            else if (owner[r] != g) { disjoint = false; break; }
        }
    }

    if (!complete) {
        std::cout << "[hmatrix] WARNING: matvec falls back to the ATOMIC schedule -- a "
                     "disk-backed block hides its row map, so row disjointness cannot be "
                     "verified. Results will vary in the last bits between runs."
                  << std::endl;
        matvec_schedule_ = MatvecSchedule::Legacy;
        return;
    }
    if (!disjoint) {
        std::cout << "[hmatrix] WARNING: matvec falls back to the ATOMIC schedule -- two "
                     "root blocks write the same row, so no thread can own a row. "
                     "Results will vary in the last bits between runs."
                  << std::endl;
        matvec_schedule_ = MatvecSchedule::Legacy;
        return;
    }

    matvec_schedule_ = grid ? MatvecSchedule::RowGroups : MatvecSchedule::PerBlock;
    matvec_grid_dim_ = grid ? N : 0;
    std::cout << "[hmatrix] matvec schedule: "
              << (grid ? "row groups" : "per block")
              << " (row disjointness verified over " << nb
              << " root blocks) -- reproducible run to run at any thread count."
              << std::endl;
}

void HMatrixViewFactor::matvec(const std::vector<double>& x,
                                std::vector<double>& y) const {
    y.assign(n_rows_, 0.0);

    // Use mmap path if loaded via mmap. NOT given the deterministic schedule
    // below: the selection threshold in radiation_exchange.cpp is set to 1e9
    // faces, so this branch is unreachable in every shipped configuration, and
    // an untested reordering of a dead production path buys nothing. If it is
    // ever re-enabled, it needs the same treatment -- its accumulation is the
    // same non-associative atomic.
    if (mmap_loaded_) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::cout << "[hmatrix] WARNING: the mmap matvec accumulates through omp "
                         "atomic; its results are not reproducible run to run."
                      << std::endl;
        }
        #pragma omp parallel for schedule(dynamic)
        for (size_t b = 0; b < mmap_root_indices_.size(); ++b) {
            matvecMmapBlock(mmap_root_indices_[b], x.data(), y.data());
        }
        return;
    }

    prepareMatvecSchedule();

    // Standard in-memory path. Each schedule below reaches a given y[i] from
    // exactly one thread, in one fixed block order, so the sum is the same
    // sum every run -- at any thread count, under any scheduling decision the
    // runtime makes.
    if (matvec_schedule_ == MatvecSchedule::RowGroups) {
        const size_t N = matvec_grid_dim_;
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                const auto& blk = root_blocks_[i * N + j];
                if (blk) blk->matvec(x.data(), y.data());
            }
        }
        return;
    }

    // PerBlock (verified: one block owns each row) and Legacy (unverified) run
    // the same loop. Only PerBlock is reproducible, and prepareMatvecSchedule()
    // has already said so for the other case.
    #pragma omp parallel for schedule(dynamic)
    for (size_t b = 0; b < root_blocks_.size(); ++b) {
        if (root_blocks_[b]) root_blocks_[b]->matvec(x.data(), y.data());
    }
}

size_t HMatrixViewFactor::memoryBytes() const {
    // For mmap'd matrices, report RAM usage (index only) not file size
    if (mmap_loaded_) {
        size_t index_memory = 0;
        for (const auto& info : mmap_blocks_) {
            index_memory += sizeof(MmapBlockInfo);
            index_memory += info.row_indices.capacity() * sizeof(int);
            index_memory += info.col_indices.capacity() * sizeof(int);
            index_memory += info.child_indices.capacity() * sizeof(size_t);
        }
        index_memory += mmap_root_indices_.capacity() * sizeof(size_t);
        return index_memory;
    }

    size_t total = 0;
    for (const auto& block : root_blocks_) {
        if (block) {
            total += block->memoryBytes();
        }
    }
    return total +
           face_centers_.size() * sizeof(std::array<double, 3>) +
           face_normals_.size() * sizeof(std::array<double, 3>) +
           face_areas_.size() * sizeof(double);
}

double HMatrixViewFactor::compressionRatio() const {
    size_t dense_size = n_rows_ * n_cols_ * sizeof(double);
    size_t actual_size = memoryBytes();
    return static_cast<double>(dense_size) / actual_size;
}

size_t HMatrixViewFactor::crossGeneratorPairs() { return g_cross_pairs.load(); }
size_t HMatrixViewFactor::crossGeneratorNearPairs() { return g_cross_near_pairs.load(); }
double HMatrixViewFactor::crossGeneratorMinROver4d() { return g_cross_min_r_over_4d.load(); }

void HMatrixViewFactor::computeDenseReference(std::vector<double>& out) {
    const size_t n = n_rows_;
    out.assign(n * n, 0.0);
    if (n == 0 || !mesh_) return;

    std::vector<int> all(n);
    for (size_t i = 0; i < n; ++i) all[i] = static_cast<int>(i);

    SparseBlock sparse;
    computeViewFactorBlock(all, all, sparse);

    for (int i = 0; i < sparse.n_rows; ++i) {
        for (int k = sparse.row_ptrs[i]; k < sparse.row_ptrs[i + 1]; ++k) {
            out[static_cast<size_t>(i) * n + static_cast<size_t>(sparse.col_indices[k])] =
                sparse.values[k];
        }
    }
}

void HMatrixViewFactor::printStatistics() const {
    std::cout << "H-matrix statistics:" << std::endl;
    std::cout << "  Dimensions: " << n_rows_ << " x " << n_cols_ << std::endl;
    std::cout << "  Block counts:" << std::endl;
    std::cout << "    Zero blocks: " << num_zero_blocks_ << std::endl;
    std::cout << "    Sparse blocks: " << num_sparse_blocks_ << std::endl;
    std::cout << "    Low-rank blocks: " << num_lowrank_blocks_ << std::endl;
    std::cout << "    Dense blocks: " << num_dense_blocks_ << std::endl;
    std::cout << "    Hierarchical blocks: " << num_hierarchical_blocks_ << std::endl;
    std::cout << "  Total non-zeros: " << total_nnz_ << std::endl;
    std::cout << "  Memory usage: " << (memoryBytes() / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "  Compression ratio: " << compressionRatio() << "x" << std::endl;
}

void HMatrixViewFactor::printContentDigest() const {
    // Provenance: two builds of the same commit on the same mesh at the same
    // thread count must print the same digest. This is the only line in the log
    // that can say so.
    const char* hash_env = std::getenv("PTM_HMATRIX_CONTENT_HASH");
    if (hash_env && std::string(hash_env) == "0") {
        std::cout << "  Content digest: DISABLED by PTM_HMATRIX_CONTENT_HASH=0 "
                     "(reproducibility of this matrix is unverified)" << std::endl;
        return;
    }
    const auto t0 = std::chrono::high_resolution_clock::now();
    const uint64_t digest = contentHash();
    const double secs = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t0).count();
    if (digest == 0) return;  // contentHash() already said why
    std::cout << "  Content digest: 0x" << std::hex << std::setw(16)
              << std::setfill('0') << digest << std::dec << std::setfill(' ')
              << "  (build threads: " << hmatrix_build_threads_
              << ", digest " << std::fixed << std::setprecision(2) << secs << " s)"
              << std::defaultfloat << std::endl;
}

// ---------------------------------------------------------------------------
// Deterministic content digest
// ---------------------------------------------------------------------------
namespace {

// FNV-1a over raw bytes. Chosen because it is order-sensitive and trivially
// auditable, not because it is cryptographic: the question this answers is
// "did the same build produce the same matrix", and any collision-resistant
// mixing does that.
inline void hashBytes(uint64_t& h, const void* p, size_t n) {
    const unsigned char* b = static_cast<const unsigned char*>(p);
    for (size_t i = 0; i < n; ++i) {
        h ^= static_cast<uint64_t>(b[i]);
        h *= 1099511628211ULL;
    }
}

inline void hashInt(uint64_t& h, int64_t v) { hashBytes(h, &v, sizeof(v)); }

inline void hashDoubles(uint64_t& h, const double* p, size_t n) {
    hashBytes(h, p, n * sizeof(double));
}

inline void hashIndices(uint64_t& h, const std::vector<int>& v) {
    hashInt(h, static_cast<int64_t>(v.size()));
    if (!v.empty()) hashBytes(h, v.data(), v.size() * sizeof(int));
}

// Returns false if the block (or a descendant) is disk-backed, in which case
// the digest is not computable from memory and the caller must say so.
bool hashBlock(uint64_t& h, const radiation::HMatrixBlock& blk) {
    hashIndices(h, blk.row_indices);
    hashIndices(h, blk.col_indices);
    hashInt(h, static_cast<int64_t>(blk.data.index()));

    if (const auto* z = std::get_if<radiation::ZeroBlock>(&blk.data)) {
        hashInt(h, z->n_rows); hashInt(h, z->n_cols);
        return true;
    }
    if (const auto* s = std::get_if<radiation::SparseBlock>(&blk.data)) {
        hashInt(h, s->n_rows); hashInt(h, s->n_cols);
        hashInt(h, static_cast<int64_t>(s->values.size()));
        if (!s->values.empty()) hashDoubles(h, s->values.data(), s->values.size());
        hashIndices(h, s->col_indices);
        hashIndices(h, s->row_ptrs);
        return true;
    }
    if (const auto* l = std::get_if<radiation::LowRankBlock>(&blk.data)) {
        hashInt(h, l->n_rows); hashInt(h, l->n_cols); hashInt(h, l->rank);
        if (l->U.size()) hashDoubles(h, l->U.data(), static_cast<size_t>(l->U.size()));
        if (l->sigma.size()) hashDoubles(h, l->sigma.data(), static_cast<size_t>(l->sigma.size()));
        if (l->V.size()) hashDoubles(h, l->V.data(), static_cast<size_t>(l->V.size()));
        return true;
    }
    if (const auto* d = std::get_if<radiation::DenseBlock>(&blk.data)) {
        hashInt(h, d->n_rows); hashInt(h, d->n_cols);
        if (d->data.size()) hashDoubles(h, d->data.data(), static_cast<size_t>(d->data.size()));
        return true;
    }
    if (const auto* hb = std::get_if<radiation::HierarchicalBlock>(&blk.data)) {
        hashInt(h, hb->n_rows); hashInt(h, hb->n_cols);
        hashIndices(h, hb->row_indices);
        hashIndices(h, hb->col_indices);
        hashInt(h, static_cast<int64_t>(hb->children.size()));
        bool ok = true;
        for (const auto& child : hb->children) {
            if (!child) { hashInt(h, -1); continue; }
            ok = hashBlock(h, *child) && ok;
        }
        return ok;
    }
    // DiskBlock: the payload lives in a file this walk does not own.
    return false;
}

} // namespace

uint64_t HMatrixViewFactor::contentHash() const {
    if (mmap_loaded_) {
        std::cout << "[hmatrix] content digest NOT computed: matrix is mmap-loaded, "
                     "its bytes are the file's. Digest the file instead." << std::endl;
        return 0;
    }
    uint64_t h = 14695981039346656037ULL;  // FNV-1a offset basis
    hashInt(h, static_cast<int64_t>(n_rows_));
    hashInt(h, static_cast<int64_t>(n_cols_));
    hashInt(h, static_cast<int64_t>(root_blocks_.size()));
    bool complete = true;
    for (size_t b = 0; b < root_blocks_.size(); ++b) {
        hashInt(h, static_cast<int64_t>(b));
        if (!root_blocks_[b]) { hashInt(h, -1); continue; }
        complete = hashBlock(h, *root_blocks_[b]) && complete;
    }
    if (!complete) {
        std::cout << "[hmatrix] content digest NOT computed: at least one block is "
                     "disk-backed (out-of-core build)." << std::endl;
        return 0;
    }
    return h;
}

// Vector utilities
std::array<double, 3> HMatrixViewFactor::subtract(
    const std::array<double, 3>& a,
    const std::array<double, 3>& b) {
    return {{a[0] - b[0], a[1] - b[1], a[2] - b[2]}};
}

double HMatrixViewFactor::dot(const std::array<double, 3>& a,
                               const std::array<double, 3>& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

double HMatrixViewFactor::length(const std::array<double, 3>& v) {
    return std::sqrt(dot(v, v));
}

std::array<double, 3> HMatrixViewFactor::cross(
    const std::array<double, 3>& a,
    const std::array<double, 3>& b) {
    return {{
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    }};
}

// ============================================================================
// Serialization Implementation
// ============================================================================

// Magic number and version for file format
static constexpr uint32_t HMATRIX_MAGIC = 0x484D4154;  // "HMAT"
// Version 2: Fixed sigma vector size type (int vs size_t mismatch)
static constexpr uint32_t HMATRIX_VERSION = 2;

// Block type enum for serialization
enum class BlockType : uint8_t {
    Zero = 0,
    Sparse = 1,
    LowRank = 2,
    Dense = 3,
    Hierarchical = 4
};

// Helper to write a vector of primitives
template<typename T>
static void writeVector(std::ostream& out, const std::vector<T>& vec) {
    size_t size = vec.size();
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));
    if (size > 0) {
        out.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(T));
    }
}

// Helper to read a vector of primitives
template<typename T>
static void readVector(std::istream& in, std::vector<T>& vec) {
    size_t size;
    in.read(reinterpret_cast<char*>(&size), sizeof(size));
    vec.resize(size);
    if (size > 0) {
        in.read(reinterpret_cast<char*>(vec.data()), size * sizeof(T));
    }
}

// Helper to write Eigen matrix
static void writeEigenMatrix(std::ostream& out, const Eigen::MatrixXd& mat) {
    int rows = static_cast<int>(mat.rows());
    int cols = static_cast<int>(mat.cols());
    out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    if (rows > 0 && cols > 0) {
        out.write(reinterpret_cast<const char*>(mat.data()), rows * cols * sizeof(double));
    }
}

// Helper to read Eigen matrix
static void readEigenMatrix(std::istream& in, Eigen::MatrixXd& mat) {
    int rows, cols;
    in.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    in.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    mat.resize(rows, cols);
    if (rows > 0 && cols > 0) {
        in.read(reinterpret_cast<char*>(mat.data()), rows * cols * sizeof(double));
    }
}

// Helper to write Eigen vector
static void writeEigenVector(std::ostream& out, const Eigen::VectorXd& vec) {
    int size = static_cast<int>(vec.size());
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));
    if (size > 0) {
        out.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(double));
    }
}

// Helper to read Eigen vector
static void readEigenVector(std::istream& in, Eigen::VectorXd& vec) {
    int size;
    in.read(reinterpret_cast<char*>(&size), sizeof(size));
    vec.resize(size);
    if (size > 0) {
        in.read(reinterpret_cast<char*>(vec.data()), size * sizeof(double));
    }
}

// Forward declaration for recursive write
static void writeBlock(std::ostream& out, const HMatrixBlock& block);

// Write individual block types
static void writeZeroBlock(std::ostream& out, const ZeroBlock& block) {
    out.write(reinterpret_cast<const char*>(&block.n_rows), sizeof(block.n_rows));
    out.write(reinterpret_cast<const char*>(&block.n_cols), sizeof(block.n_cols));
}

static void writeSparseBlock(std::ostream& out, const SparseBlock& block) {
    out.write(reinterpret_cast<const char*>(&block.n_rows), sizeof(block.n_rows));
    out.write(reinterpret_cast<const char*>(&block.n_cols), sizeof(block.n_cols));
    writeVector(out, block.values);
    writeVector(out, block.col_indices);
    writeVector(out, block.row_ptrs);
}

static void writeLowRankBlock(std::ostream& out, const LowRankBlock& block) {
    out.write(reinterpret_cast<const char*>(&block.n_rows), sizeof(block.n_rows));
    out.write(reinterpret_cast<const char*>(&block.n_cols), sizeof(block.n_cols));
    out.write(reinterpret_cast<const char*>(&block.rank), sizeof(block.rank));
    writeEigenMatrix(out, block.U);
    writeEigenVector(out, block.sigma);
    writeEigenMatrix(out, block.V);
}

static void writeDenseBlock(std::ostream& out, const DenseBlock& block) {
    out.write(reinterpret_cast<const char*>(&block.n_rows), sizeof(block.n_rows));
    out.write(reinterpret_cast<const char*>(&block.n_cols), sizeof(block.n_cols));
    writeEigenMatrix(out, block.data);
}

static void writeHierarchicalBlock(std::ostream& out, const HierarchicalBlock& block) {
    out.write(reinterpret_cast<const char*>(&block.n_rows), sizeof(block.n_rows));
    out.write(reinterpret_cast<const char*>(&block.n_cols), sizeof(block.n_cols));
    writeVector(out, block.row_indices);
    writeVector(out, block.col_indices);

    // Write number of children
    size_t num_children = block.children.size();
    out.write(reinterpret_cast<const char*>(&num_children), sizeof(num_children));

    // Write each child recursively
    for (const auto& child : block.children) {
        bool has_child = (child != nullptr);
        out.write(reinterpret_cast<const char*>(&has_child), sizeof(has_child));
        if (has_child) {
            writeBlock(out, *child);
        }
    }
}

static void writeBlock(std::ostream& out, const HMatrixBlock& block) {
    // Write row/col indices
    writeVector(out, block.row_indices);
    writeVector(out, block.col_indices);

    // Write block type and data
    BlockType type;
    if (std::holds_alternative<ZeroBlock>(block.data)) {
        type = BlockType::Zero;
        out.write(reinterpret_cast<const char*>(&type), sizeof(type));
        writeZeroBlock(out, std::get<ZeroBlock>(block.data));
    } else if (std::holds_alternative<SparseBlock>(block.data)) {
        type = BlockType::Sparse;
        out.write(reinterpret_cast<const char*>(&type), sizeof(type));
        writeSparseBlock(out, std::get<SparseBlock>(block.data));
    } else if (std::holds_alternative<LowRankBlock>(block.data)) {
        type = BlockType::LowRank;
        out.write(reinterpret_cast<const char*>(&type), sizeof(type));
        writeLowRankBlock(out, std::get<LowRankBlock>(block.data));
    } else if (std::holds_alternative<DenseBlock>(block.data)) {
        type = BlockType::Dense;
        out.write(reinterpret_cast<const char*>(&type), sizeof(type));
        writeDenseBlock(out, std::get<DenseBlock>(block.data));
    } else if (std::holds_alternative<HierarchicalBlock>(block.data)) {
        type = BlockType::Hierarchical;
        out.write(reinterpret_cast<const char*>(&type), sizeof(type));
        writeHierarchicalBlock(out, std::get<HierarchicalBlock>(block.data));
    } else if (std::holds_alternative<DiskBlock>(block.data)) {
        // DiskBlock: reconstruct as SparseBlock from mmap data and write
        const auto& disk = std::get<DiskBlock>(block.data);

        if (disk.block_type == 0) {
            // Zero block
            type = BlockType::Zero;
            out.write(reinterpret_cast<const char*>(&type), sizeof(type));
            ZeroBlock zero;
            zero.n_rows = disk.n_rows;
            zero.n_cols = disk.n_cols;
            writeZeroBlock(out, zero);
        } else if (disk.block_type == 1 && disk.mmap_base) {
            // Sparse block - read from mmap
            type = BlockType::Sparse;
            out.write(reinterpret_cast<const char*>(&type), sizeof(type));

            const char* data_ptr = disk.mmap_base + disk.file_offset;

            // Skip the indices stored in the disk file (we already wrote them above)
            data_ptr += block.row_indices.size() * sizeof(int);
            data_ptr += block.col_indices.size() * sizeof(int);

            // Read sparse block data from mmap
            SparseBlock sparse;
            sparse.n_rows = disk.n_rows;
            sparse.n_cols = disk.n_cols;

            sparse.row_ptrs.resize(disk.n_rows + 1);
            std::memcpy(sparse.row_ptrs.data(), data_ptr, (disk.n_rows + 1) * sizeof(int));
            data_ptr += (disk.n_rows + 1) * sizeof(int);

            sparse.col_indices.resize(disk.nnz);
            std::memcpy(sparse.col_indices.data(), data_ptr, disk.nnz * sizeof(int));
            data_ptr += disk.nnz * sizeof(int);

            sparse.values.resize(disk.nnz);
            std::memcpy(sparse.values.data(), data_ptr, disk.nnz * sizeof(double));

            writeSparseBlock(out, sparse);
        } else if (disk.block_type == 2 && disk.mmap_base) {
            // LowRank block - read from mmap
            type = BlockType::LowRank;
            out.write(reinterpret_cast<const char*>(&type), sizeof(type));

            const char* data_ptr = disk.mmap_base + disk.file_offset;

            // Skip the indices stored in the disk file
            data_ptr += block.row_indices.size() * sizeof(int);
            data_ptr += block.col_indices.size() * sizeof(int);

            LowRankBlock lowrank;
            lowrank.n_rows = disk.n_rows;
            lowrank.n_cols = disk.n_cols;
            lowrank.rank = disk.rank;

            lowrank.U.resize(disk.n_rows, disk.rank);
            std::memcpy(lowrank.U.data(), data_ptr, disk.n_rows * disk.rank * sizeof(double));
            data_ptr += disk.n_rows * disk.rank * sizeof(double);

            lowrank.sigma.resize(disk.rank);
            std::memcpy(lowrank.sigma.data(), data_ptr, disk.rank * sizeof(double));
            data_ptr += disk.rank * sizeof(double);

            lowrank.V.resize(disk.n_cols, disk.rank);
            std::memcpy(lowrank.V.data(), data_ptr, disk.n_cols * disk.rank * sizeof(double));

            writeLowRankBlock(out, lowrank);
        } else {
            // Fallback: write as zero block if mmap not available
            type = BlockType::Zero;
            out.write(reinterpret_cast<const char*>(&type), sizeof(type));
            ZeroBlock zero;
            zero.n_rows = disk.n_rows;
            zero.n_cols = disk.n_cols;
            writeZeroBlock(out, zero);
        }
    }
}

// Forward declaration for recursive read
static std::unique_ptr<HMatrixBlock> readBlock(std::istream& in);

// Read individual block types
static ZeroBlock readZeroBlock(std::istream& in) {
    ZeroBlock block;
    in.read(reinterpret_cast<char*>(&block.n_rows), sizeof(block.n_rows));
    in.read(reinterpret_cast<char*>(&block.n_cols), sizeof(block.n_cols));
    return block;
}

static SparseBlock readSparseBlock(std::istream& in) {
    SparseBlock block;
    in.read(reinterpret_cast<char*>(&block.n_rows), sizeof(block.n_rows));
    in.read(reinterpret_cast<char*>(&block.n_cols), sizeof(block.n_cols));
    readVector(in, block.values);
    readVector(in, block.col_indices);
    readVector(in, block.row_ptrs);
    return block;
}

static LowRankBlock readLowRankBlock(std::istream& in) {
    LowRankBlock block;
    in.read(reinterpret_cast<char*>(&block.n_rows), sizeof(block.n_rows));
    in.read(reinterpret_cast<char*>(&block.n_cols), sizeof(block.n_cols));
    in.read(reinterpret_cast<char*>(&block.rank), sizeof(block.rank));
    readEigenMatrix(in, block.U);
    readEigenVector(in, block.sigma);
    readEigenMatrix(in, block.V);
    return block;
}

static DenseBlock readDenseBlock(std::istream& in) {
    DenseBlock block;
    in.read(reinterpret_cast<char*>(&block.n_rows), sizeof(block.n_rows));
    in.read(reinterpret_cast<char*>(&block.n_cols), sizeof(block.n_cols));
    readEigenMatrix(in, block.data);
    return block;
}

static HierarchicalBlock readHierarchicalBlock(std::istream& in) {
    HierarchicalBlock block;
    in.read(reinterpret_cast<char*>(&block.n_rows), sizeof(block.n_rows));
    in.read(reinterpret_cast<char*>(&block.n_cols), sizeof(block.n_cols));
    readVector(in, block.row_indices);
    readVector(in, block.col_indices);

    // Read number of children
    size_t num_children;
    in.read(reinterpret_cast<char*>(&num_children), sizeof(num_children));
    block.children.resize(num_children);

    // Read each child recursively
    for (size_t c = 0; c < num_children; ++c) {
        bool has_child;
        in.read(reinterpret_cast<char*>(&has_child), sizeof(has_child));
        if (has_child) {
            block.children[c] = readBlock(in);
        }
    }
    return block;
}

static std::unique_ptr<HMatrixBlock> readBlock(std::istream& in) {
    auto block = std::make_unique<HMatrixBlock>();

    // Read row/col indices
    readVector(in, block->row_indices);
    readVector(in, block->col_indices);

    // Read block type
    BlockType type;
    in.read(reinterpret_cast<char*>(&type), sizeof(type));

    switch (type) {
        case BlockType::Zero:
            block->data = readZeroBlock(in);
            break;
        case BlockType::Sparse:
            block->data = readSparseBlock(in);
            break;
        case BlockType::LowRank:
            block->data = readLowRankBlock(in);
            break;
        case BlockType::Dense:
            block->data = readDenseBlock(in);
            break;
        case BlockType::Hierarchical:
            block->data = readHierarchicalBlock(in);
            break;
        default:
            throw std::runtime_error("Unknown block type in H-matrix file");
    }

    return block;
}

// ============================================================================
// Out-of-core support - write blocks to disk during build to save RAM
// ============================================================================

bool HMatrixViewFactor::initOutOfCore(const std::string& path) {
    out_of_core_file_ = path + ".hmat.tmp";

    // Open file for writing (create/truncate)
    out_of_core_fd_ = open(out_of_core_file_.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (out_of_core_fd_ < 0) {
        std::cerr << "Failed to create out-of-core file: " << out_of_core_file_ << std::endl;
        return false;
    }

    out_of_core_enabled_ = true;
    out_of_core_size_ = 0;

#ifdef __APPLE__
    // macOS: bypass the unified buffer cache for streamed block writes.
    // Without this, write() pages accumulate in the UBC (not counted as
    // free+inactive by getAvailableMemory) and jetsam OOM-kills the build
    // on large meshes. F_FULLFSYNC alone does not evict these pages.
    fcntl(out_of_core_fd_, F_NOCACHE, 1);
    std::cout << "  Out-of-core storage enabled (direct/uncached I/O): "
              << out_of_core_file_ << std::endl;
#else
    std::cout << "  Out-of-core storage enabled: " << out_of_core_file_ << std::endl;
#endif
    return true;
}

void HMatrixViewFactor::finalizeOutOfCore() {
    if (!out_of_core_enabled_ || out_of_core_fd_ < 0) return;

    // Sync all writes
    fsync(out_of_core_fd_);

    // Get final file size
    out_of_core_size_ = lseek(out_of_core_fd_, 0, SEEK_END);

#ifdef __APPLE__
    // Restore normal caching for the read phase: repeated matvec reads during
    // the radiosity solve benefit from the page cache, and these are clean,
    // reclaimable pages (not the build-time write accumulation we guarded
    // against in initOutOfCore).
    fcntl(out_of_core_fd_, F_NOCACHE, 0);
#endif

    // Memory-map the file for reading
    if (out_of_core_size_ > 0) {
        out_of_core_mmap_ = static_cast<char*>(mmap(
            nullptr, out_of_core_size_, PROT_READ, MAP_SHARED, out_of_core_fd_, 0));

        if (out_of_core_mmap_ == MAP_FAILED) {
            std::cerr << "Warning: Failed to mmap out-of-core file, will use read() instead" << std::endl;
            out_of_core_mmap_ = nullptr;
        } else {
            // Advise kernel we'll access sequentially during matvec
            madvise(out_of_core_mmap_, out_of_core_size_, MADV_SEQUENTIAL);
        }
    }

    std::cout << "  Out-of-core file size: " << std::fixed << std::setprecision(2)
              << (out_of_core_size_ / (1024.0 * 1024.0)) << " MB" << std::endl;
}

void HMatrixViewFactor::cleanupOutOfCore() {
    if (out_of_core_mmap_ && out_of_core_mmap_ != MAP_FAILED) {
        munmap(out_of_core_mmap_, out_of_core_size_);
        out_of_core_mmap_ = nullptr;
    }

    if (out_of_core_fd_ >= 0) {
        close(out_of_core_fd_);
        out_of_core_fd_ = -1;
    }

    // Delete temporary file
    if (!out_of_core_file_.empty()) {
        unlink(out_of_core_file_.c_str());
        out_of_core_file_.clear();
    }

    out_of_core_enabled_ = false;
    out_of_core_size_ = 0;
}

DiskBlock HMatrixViewFactor::writeBlockToDisk(
    const SparseBlock& sparse,
    const std::vector<int>& row_indices,
    const std::vector<int>& col_indices) {

    DiskBlock disk;
    disk.n_rows = sparse.n_rows;
    disk.n_cols = sparse.n_cols;
    disk.block_type = 1;  // Sparse
    disk.nnz = static_cast<int>(sparse.values.size());
    disk.rank = 0;

    // Calculate total data size:
    // - row_indices: n_rows * sizeof(int)
    // - col_indices: n_cols * sizeof(int)
    // - row_ptrs: (n_rows + 1) * sizeof(int)
    // - col_indices (sparse): nnz * sizeof(int)
    // - values: nnz * sizeof(double)
    size_t data_size =
        row_indices.size() * sizeof(int) +
        col_indices.size() * sizeof(int) +
        sparse.row_ptrs.size() * sizeof(int) +
        sparse.col_indices.size() * sizeof(int) +
        sparse.values.size() * sizeof(double);

    disk.data_size = static_cast<int64_t>(data_size);

    // Write to file (thread-safe)
    {
        std::lock_guard<std::mutex> lock(out_of_core_mutex_);

        disk.file_offset = lseek(out_of_core_fd_, 0, SEEK_END);

        // Write all data contiguously
        write(out_of_core_fd_, row_indices.data(), row_indices.size() * sizeof(int));
        write(out_of_core_fd_, col_indices.data(), col_indices.size() * sizeof(int));
        write(out_of_core_fd_, sparse.row_ptrs.data(), sparse.row_ptrs.size() * sizeof(int));
        write(out_of_core_fd_, sparse.col_indices.data(), sparse.col_indices.size() * sizeof(int));
        write(out_of_core_fd_, sparse.values.data(), sparse.values.size() * sizeof(double));
    }

    return disk;
}

DiskBlock HMatrixViewFactor::writeBlockToDisk(
    const LowRankBlock& lowrank,
    const std::vector<int>& row_indices,
    const std::vector<int>& col_indices) {

    DiskBlock disk;
    disk.n_rows = lowrank.n_rows;
    disk.n_cols = lowrank.n_cols;
    disk.block_type = 2;  // LowRank
    disk.nnz = 0;
    disk.rank = lowrank.rank;

    // Calculate total data size:
    // - row_indices: n_rows * sizeof(int)
    // - col_indices: n_cols * sizeof(int)
    // - U: n_rows * rank * sizeof(double)
    // - sigma: rank * sizeof(double)
    // - V: n_cols * rank * sizeof(double)
    size_t data_size =
        row_indices.size() * sizeof(int) +
        col_indices.size() * sizeof(int) +
        lowrank.U.size() * sizeof(double) +
        lowrank.sigma.size() * sizeof(double) +
        lowrank.V.size() * sizeof(double);

    disk.data_size = static_cast<int64_t>(data_size);

    // Write to file (thread-safe)
    {
        std::lock_guard<std::mutex> lock(out_of_core_mutex_);

        disk.file_offset = lseek(out_of_core_fd_, 0, SEEK_END);

        // Write indices
        write(out_of_core_fd_, row_indices.data(), row_indices.size() * sizeof(int));
        write(out_of_core_fd_, col_indices.data(), col_indices.size() * sizeof(int));

        // Write matrix data (column-major for Eigen)
        write(out_of_core_fd_, lowrank.U.data(), lowrank.U.size() * sizeof(double));
        write(out_of_core_fd_, lowrank.sigma.data(), lowrank.sigma.size() * sizeof(double));
        write(out_of_core_fd_, lowrank.V.data(), lowrank.V.size() * sizeof(double));
    }

    return disk;
}

// DiskBlock::matvec - read data AND indices from disk and compute
// This version reads indices from the mmap file to save RAM (memory-efficient out-of-core)
void DiskBlock::matvec(const double* x, double* y) const {
    if (block_type == 0) return;  // Zero block

    if (!mmap_base) {
        std::cerr << "DiskBlock::matvec called without mmap_base set!" << std::endl;
        return;
    }

    const char* data_ptr = mmap_base + file_offset;

    // Read indices from mmap file (instead of keeping in RAM)
    const int* row_indices = reinterpret_cast<const int*>(data_ptr);
    data_ptr += n_rows * sizeof(int);

    const int* col_indices = reinterpret_cast<const int*>(data_ptr);
    data_ptr += n_cols * sizeof(int);

    if (block_type == 1) {
        // Sparse block
        const int* row_ptrs = reinterpret_cast<const int*>(data_ptr);
        data_ptr += (n_rows + 1) * sizeof(int);

        const int* sparse_col_indices = reinterpret_cast<const int*>(data_ptr);
        data_ptr += nnz * sizeof(int);

        const double* values = reinterpret_cast<const double*>(data_ptr);

        // CSR SpMV: y[global_row] += sum_j(A[i,j] * x[global_col_j])
        for (int i = 0; i < n_rows; ++i) {
            int global_row = row_indices[i];
            double sum = 0.0;

            for (int k = row_ptrs[i]; k < row_ptrs[i + 1]; ++k) {
                int local_col = sparse_col_indices[k];
                int global_col = col_indices[local_col];
                sum += values[k] * x[global_col];
            }

            #pragma omp atomic
            y[global_row] += sum;
        }
    } else if (block_type == 2) {
        // Low-rank block: y += U * diag(sigma) * V^T * x
        const double* U_data = reinterpret_cast<const double*>(data_ptr);
        data_ptr += n_rows * rank * sizeof(double);

        const double* sigma_data = reinterpret_cast<const double*>(data_ptr);
        data_ptr += rank * sizeof(double);

        const double* V_data = reinterpret_cast<const double*>(data_ptr);

        // Step 1: temp = V^T * x (gather from global x using col_indices)
        std::vector<double> temp(rank, 0.0);
        for (int r = 0; r < rank; ++r) {
            double sum = 0.0;
            for (int j = 0; j < n_cols; ++j) {
                int global_col = col_indices[j];
                sum += V_data[j + r * n_cols] * x[global_col];  // V is col-major
            }
            temp[r] = sigma_data[r] * sum;
        }

        // Step 2: y += U * temp (scatter to global y using row_indices)
        for (int i = 0; i < n_rows; ++i) {
            int global_row = row_indices[i];
            double sum = 0.0;
            for (int r = 0; r < rank; ++r) {
                sum += U_data[i + r * n_rows] * temp[r];  // U is col-major
            }
            #pragma omp atomic
            y[global_row] += sum;
        }
    }
}

bool HMatrixViewFactor::writeBinary(const std::string& filename) const {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return false;
    }

    // Use larger buffer for faster I/O
    constexpr size_t BUFFER_SIZE = 16 * 1024 * 1024;  // 16 MB buffer
    std::vector<char> buffer(BUFFER_SIZE);
    out.rdbuf()->pubsetbuf(buffer.data(), BUFFER_SIZE);

    std::cout << "Saving H-matrix to " << filename << "..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    try {
        // Write header
        out.write(reinterpret_cast<const char*>(&HMATRIX_MAGIC), sizeof(HMATRIX_MAGIC));
        out.write(reinterpret_cast<const char*>(&HMATRIX_VERSION), sizeof(HMATRIX_VERSION));

        // Write dimensions
        out.write(reinterpret_cast<const char*>(&n_rows_), sizeof(n_rows_));
        out.write(reinterpret_cast<const char*>(&n_cols_), sizeof(n_cols_));

        // Write number of root blocks
        size_t num_blocks = root_blocks_.size();
        out.write(reinterpret_cast<const char*>(&num_blocks), sizeof(num_blocks));

        std::cout << "  Writing " << num_blocks << " root blocks..." << std::flush;

        // Progress tracking
        size_t blocks_written = 0;
        size_t last_percent = 0;

        // Write each root block
        for (const auto& block : root_blocks_) {
            bool has_block = (block != nullptr);
            out.write(reinterpret_cast<const char*>(&has_block), sizeof(has_block));
            if (has_block) {
                writeBlock(out, *block);
            }

            // Progress update
            blocks_written++;
            size_t percent = (blocks_written * 100) / num_blocks;
            if (percent > last_percent && percent % 10 == 0) {
                std::cout << " " << percent << "%" << std::flush;
                last_percent = percent;
            }
        }
        std::cout << " done" << std::endl;

        // Write statistics for verification
        out.write(reinterpret_cast<const char*>(&num_zero_blocks_), sizeof(num_zero_blocks_));
        out.write(reinterpret_cast<const char*>(&num_sparse_blocks_), sizeof(num_sparse_blocks_));
        out.write(reinterpret_cast<const char*>(&num_lowrank_blocks_), sizeof(num_lowrank_blocks_));
        out.write(reinterpret_cast<const char*>(&num_dense_blocks_), sizeof(num_dense_blocks_));
        out.write(reinterpret_cast<const char*>(&num_hierarchical_blocks_), sizeof(num_hierarchical_blocks_));
        out.write(reinterpret_cast<const char*>(&total_nnz_), sizeof(total_nnz_));

        out.flush();

        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end_time - start_time).count();

        std::cout << "H-matrix saved to " << filename << " (" << out.tellp() << " bytes) in "
                  << std::fixed << std::setprecision(1) << elapsed << " seconds" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error writing H-matrix: " << e.what() << std::endl;
        return false;
    }
}

bool HMatrixViewFactor::readBinary(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
        return false;
    }

    try {
        // Read and verify header
        uint32_t magic, version;
        in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        in.read(reinterpret_cast<char*>(&version), sizeof(version));

        if (magic != HMATRIX_MAGIC) {
            std::cerr << "Invalid H-matrix file format (bad magic number)" << std::endl;
            return false;
        }
        if (version != HMATRIX_VERSION) {
            std::cerr << "Unsupported H-matrix file version: " << version << std::endl;
            return false;
        }

        // Read dimensions
        in.read(reinterpret_cast<char*>(&n_rows_), sizeof(n_rows_));
        in.read(reinterpret_cast<char*>(&n_cols_), sizeof(n_cols_));

        // Read number of root blocks
        size_t num_blocks;
        in.read(reinterpret_cast<char*>(&num_blocks), sizeof(num_blocks));
        root_blocks_.resize(num_blocks);

        // Read each root block
        for (size_t b = 0; b < num_blocks; ++b) {
            bool has_block;
            in.read(reinterpret_cast<char*>(&has_block), sizeof(has_block));
            if (has_block) {
                root_blocks_[b] = readBlock(in);
            }
        }

        // Read statistics
        in.read(reinterpret_cast<char*>(&num_zero_blocks_), sizeof(num_zero_blocks_));
        in.read(reinterpret_cast<char*>(&num_sparse_blocks_), sizeof(num_sparse_blocks_));
        in.read(reinterpret_cast<char*>(&num_lowrank_blocks_), sizeof(num_lowrank_blocks_));
        in.read(reinterpret_cast<char*>(&num_dense_blocks_), sizeof(num_dense_blocks_));
        in.read(reinterpret_cast<char*>(&num_hierarchical_blocks_), sizeof(num_hierarchical_blocks_));
        in.read(reinterpret_cast<char*>(&total_nnz_), sizeof(total_nnz_));

        std::cout << "H-matrix loaded from " << filename << std::endl;
        printStatistics();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error reading H-matrix: " << e.what() << std::endl;
        return false;
    }
}

// ============================================================================
// Memory-mapped loading implementation
// ============================================================================

HMatrixViewFactor::~HMatrixViewFactor() {
    cleanupOutOfCore();
    cleanupMmap();
}

HMatrixViewFactor::HMatrixViewFactor(HMatrixViewFactor&& other) noexcept
    : mesh_(std::move(other.mesh_))
    , device_(other.device_)
    , scene_(other.scene_)
    , config_(std::move(other.config_))
    , rad_config_(std::move(other.rad_config_))
    , n_rows_(other.n_rows_)
    , n_cols_(other.n_cols_)
    , root_blocks_(std::move(other.root_blocks_))
    , face_centers_(std::move(other.face_centers_))
    , face_normals_(std::move(other.face_normals_))
    , face_areas_(std::move(other.face_areas_))
    , num_zero_blocks_(other.num_zero_blocks_)
    , num_sparse_blocks_(other.num_sparse_blocks_)
    , num_lowrank_blocks_(other.num_lowrank_blocks_)
    , num_dense_blocks_(other.num_dense_blocks_)
    , num_hierarchical_blocks_(other.num_hierarchical_blocks_)
    , num_disk_blocks_(other.num_disk_blocks_)
    , total_nnz_(other.total_nnz_)
    , out_of_core_enabled_(other.out_of_core_enabled_)
    , out_of_core_file_(std::move(other.out_of_core_file_))
    , out_of_core_fd_(other.out_of_core_fd_)
    , out_of_core_mmap_(other.out_of_core_mmap_)
    , out_of_core_size_(other.out_of_core_size_)
    , mmap_loaded_(other.mmap_loaded_)
    , mmap_file_(std::move(other.mmap_file_))
    , mmap_fd_(other.mmap_fd_)
    , mmap_base_(other.mmap_base_)
    , mmap_size_(other.mmap_size_)
    , mmap_blocks_(std::move(other.mmap_blocks_))
    , mmap_root_indices_(std::move(other.mmap_root_indices_))
{
    // Clear source to prevent double-free
    other.out_of_core_fd_ = -1;
    other.out_of_core_mmap_ = nullptr;
    other.mmap_fd_ = -1;
    other.mmap_base_ = nullptr;
}

HMatrixViewFactor& HMatrixViewFactor::operator=(HMatrixViewFactor&& other) noexcept {
    if (this != &other) {
        // Clean up existing resources
        cleanupOutOfCore();
        cleanupMmap();

        // Move everything
        mesh_ = std::move(other.mesh_);
        device_ = other.device_;
        scene_ = other.scene_;
        config_ = std::move(other.config_);
        rad_config_ = std::move(other.rad_config_);
        n_rows_ = other.n_rows_;
        n_cols_ = other.n_cols_;
        root_blocks_ = std::move(other.root_blocks_);
        face_centers_ = std::move(other.face_centers_);
        face_normals_ = std::move(other.face_normals_);
        face_areas_ = std::move(other.face_areas_);
        num_zero_blocks_ = other.num_zero_blocks_;
        num_sparse_blocks_ = other.num_sparse_blocks_;
        num_lowrank_blocks_ = other.num_lowrank_blocks_;
        num_dense_blocks_ = other.num_dense_blocks_;
        num_hierarchical_blocks_ = other.num_hierarchical_blocks_;
        num_disk_blocks_ = other.num_disk_blocks_;
        total_nnz_ = other.total_nnz_;
        out_of_core_enabled_ = other.out_of_core_enabled_;
        out_of_core_file_ = std::move(other.out_of_core_file_);
        out_of_core_fd_ = other.out_of_core_fd_;
        out_of_core_mmap_ = other.out_of_core_mmap_;
        out_of_core_size_ = other.out_of_core_size_;
        mmap_loaded_ = other.mmap_loaded_;
        mmap_file_ = std::move(other.mmap_file_);
        mmap_fd_ = other.mmap_fd_;
        mmap_base_ = other.mmap_base_;
        mmap_size_ = other.mmap_size_;
        mmap_blocks_ = std::move(other.mmap_blocks_);
        mmap_root_indices_ = std::move(other.mmap_root_indices_);

        // Clear source
        other.out_of_core_fd_ = -1;
        other.out_of_core_mmap_ = nullptr;
        other.mmap_fd_ = -1;
        other.mmap_base_ = nullptr;
    }
    return *this;
}

void HMatrixViewFactor::cleanupMmap() {
    if (mmap_base_ && mmap_base_ != MAP_FAILED) {
        munmap(mmap_base_, mmap_size_);
        mmap_base_ = nullptr;
    }
    if (mmap_fd_ >= 0) {
        close(mmap_fd_);
        mmap_fd_ = -1;
    }
    mmap_loaded_ = false;
    mmap_size_ = 0;
    mmap_file_.clear();
    mmap_blocks_.clear();
    mmap_root_indices_.clear();
}

// Helper function to scan through file and build block index
// This reads just the structure (indices, types, sizes) without loading actual data
bool HMatrixViewFactor::scanBlockForMmap(
    const char* data, size_t file_size, size_t& offset,
    size_t& out_block_idx)
{
    // Helper macro for bounds checking with diagnostics
    #define MMAP_CHECK_BOUNDS(needed, msg) \
        if (offset + (needed) > file_size) { \
            std::cerr << "  [mmap scan] " << msg << ": need " << (needed) \
                      << " bytes at offset " << offset << " but file_size=" << file_size \
                      << " (block started at " << block_start_offset << ")" << std::endl; \
            return false; \
        }

    if (offset >= file_size) {
        std::cerr << "  [mmap scan] offset " << offset << " >= file_size " << file_size << std::endl;
        return false;
    }

    size_t block_start_offset = offset;
    MmapBlockInfo info;

    // Read row_indices
    // Read row_indices
    MMAP_CHECK_BOUNDS(sizeof(size_t), "row_indices length");
    size_t num_row_indices = *reinterpret_cast<const size_t*>(data + offset);
    offset += sizeof(size_t);

    MMAP_CHECK_BOUNDS(num_row_indices * sizeof(int), "row_indices data");
    info.row_indices.resize(num_row_indices);
    std::memcpy(info.row_indices.data(), data + offset, num_row_indices * sizeof(int));
    offset += num_row_indices * sizeof(int);

    // Read col_indices
    MMAP_CHECK_BOUNDS(sizeof(size_t), "col_indices length");
    size_t num_col_indices = *reinterpret_cast<const size_t*>(data + offset);
    offset += sizeof(size_t);

    MMAP_CHECK_BOUNDS(num_col_indices * sizeof(int), "col_indices data");
    info.col_indices.resize(num_col_indices);
    std::memcpy(info.col_indices.data(), data + offset, num_col_indices * sizeof(int));
    offset += num_col_indices * sizeof(int);

    // Read block type
    MMAP_CHECK_BOUNDS(sizeof(BlockType), "block type");
    BlockType type = *reinterpret_cast<const BlockType*>(data + offset);
    offset += sizeof(BlockType);

    info.block_type = static_cast<int>(type);
    info.file_offset = static_cast<int64_t>(offset);  // Data starts here

    switch (type) {
        case BlockType::Zero: {
            // Read n_rows, n_cols
            MMAP_CHECK_BOUNDS(2 * sizeof(int), "Zero n_rows/n_cols");
            info.n_rows = *reinterpret_cast<const int*>(data + offset);
            offset += sizeof(int);
            info.n_cols = *reinterpret_cast<const int*>(data + offset);
            offset += sizeof(int);
            break;
        }
        case BlockType::Sparse: {
            // Read n_rows, n_cols
            MMAP_CHECK_BOUNDS(2 * sizeof(int), "Sparse n_rows/n_cols");
            info.n_rows = *reinterpret_cast<const int*>(data + offset);
            offset += sizeof(int);
            info.n_cols = *reinterpret_cast<const int*>(data + offset);
            offset += sizeof(int);

            // Read values vector size (writeVector uses size_t for length)
            MMAP_CHECK_BOUNDS(sizeof(size_t), "Sparse values length");
            size_t num_values = *reinterpret_cast<const size_t*>(data + offset);
            offset += sizeof(size_t);
            info.nnz = static_cast<int>(num_values);

            // Skip values data
            if (num_values > 0) {
                MMAP_CHECK_BOUNDS(num_values * sizeof(double), "Sparse values data");
                offset += num_values * sizeof(double);
            }

            // Read col_indices size and skip
            MMAP_CHECK_BOUNDS(sizeof(size_t), "Sparse col_indices length");
            size_t num_col_idx = *reinterpret_cast<const size_t*>(data + offset);
            offset += sizeof(size_t);
            if (num_col_idx > 0) {
                MMAP_CHECK_BOUNDS(num_col_idx * sizeof(int), "Sparse col_indices data");
                offset += num_col_idx * sizeof(int);
            }

            // Read row_ptrs size and skip
            MMAP_CHECK_BOUNDS(sizeof(size_t), "Sparse row_ptrs length");
            size_t num_row_ptrs = *reinterpret_cast<const size_t*>(data + offset);
            offset += sizeof(size_t);
            if (num_row_ptrs > 0) {
                MMAP_CHECK_BOUNDS(num_row_ptrs * sizeof(int), "Sparse row_ptrs data");
                offset += num_row_ptrs * sizeof(int);
            }
            break;
        }
        case BlockType::LowRank: {
            // Read n_rows, n_cols, rank
            MMAP_CHECK_BOUNDS(3 * sizeof(int), "LowRank n_rows/n_cols/rank");
            info.n_rows = *reinterpret_cast<const int*>(data + offset);
            offset += sizeof(int);
            info.n_cols = *reinterpret_cast<const int*>(data + offset);
            offset += sizeof(int);
            info.rank = *reinterpret_cast<const int*>(data + offset);
            offset += sizeof(int);

            // Skip U matrix (rows, cols as int, then data)
            MMAP_CHECK_BOUNDS(2 * sizeof(int), "LowRank U dims");
            int u_rows = *reinterpret_cast<const int*>(data + offset);
            offset += sizeof(int);
            int u_cols = *reinterpret_cast<const int*>(data + offset);
            offset += sizeof(int);
            if (u_rows > 0 && u_cols > 0) {
                MMAP_CHECK_BOUNDS(static_cast<size_t>(u_rows) * u_cols * sizeof(double), "LowRank U data");
                offset += static_cast<size_t>(u_rows) * u_cols * sizeof(double);
            }

            // Skip sigma vector (writeEigenVector uses int for size, not size_t!)
            MMAP_CHECK_BOUNDS(sizeof(int), "LowRank sigma length");
            int sigma_size = *reinterpret_cast<const int*>(data + offset);
            offset += sizeof(int);
            if (sigma_size > 0) {
                MMAP_CHECK_BOUNDS(static_cast<size_t>(sigma_size) * sizeof(double), "LowRank sigma data");
                offset += static_cast<size_t>(sigma_size) * sizeof(double);
            }

            // Skip V matrix
            MMAP_CHECK_BOUNDS(2 * sizeof(int), "LowRank V dims");
            int v_rows = *reinterpret_cast<const int*>(data + offset);
            offset += sizeof(int);
            int v_cols = *reinterpret_cast<const int*>(data + offset);
            offset += sizeof(int);
            if (v_rows > 0 && v_cols > 0) {
                MMAP_CHECK_BOUNDS(static_cast<size_t>(v_rows) * v_cols * sizeof(double), "LowRank V data");
                offset += static_cast<size_t>(v_rows) * v_cols * sizeof(double);
            }
            break;
        }
        case BlockType::Dense: {
            // Read n_rows, n_cols
            MMAP_CHECK_BOUNDS(2 * sizeof(int), "Dense n_rows/n_cols");
            info.n_rows = *reinterpret_cast<const int*>(data + offset);
            offset += sizeof(int);
            info.n_cols = *reinterpret_cast<const int*>(data + offset);
            offset += sizeof(int);

            // Skip data matrix (writeEigenMatrix uses int for rows/cols)
            MMAP_CHECK_BOUNDS(2 * sizeof(int), "Dense data dims");
            int d_rows = *reinterpret_cast<const int*>(data + offset);
            offset += sizeof(int);
            int d_cols = *reinterpret_cast<const int*>(data + offset);
            offset += sizeof(int);
            if (d_rows > 0 && d_cols > 0) {
                MMAP_CHECK_BOUNDS(static_cast<size_t>(d_rows) * d_cols * sizeof(double), "Dense data");
                offset += static_cast<size_t>(d_rows) * d_cols * sizeof(double);
            }
            break;
        }
        case BlockType::Hierarchical: {
            // Read n_rows, n_cols
            MMAP_CHECK_BOUNDS(2 * sizeof(int), "Hierarchical n_rows/n_cols");
            info.n_rows = *reinterpret_cast<const int*>(data + offset);
            offset += sizeof(int);
            info.n_cols = *reinterpret_cast<const int*>(data + offset);
            offset += sizeof(int);

            // Skip row_indices (writeVector uses size_t for length)
            MMAP_CHECK_BOUNDS(sizeof(size_t), "Hierarchical row_indices length");
            size_t hier_row_idx = *reinterpret_cast<const size_t*>(data + offset);
            offset += sizeof(size_t);
            if (hier_row_idx > 0) {
                MMAP_CHECK_BOUNDS(hier_row_idx * sizeof(int), "Hierarchical row_indices data");
                offset += hier_row_idx * sizeof(int);
            }

            // Skip col_indices
            MMAP_CHECK_BOUNDS(sizeof(size_t), "Hierarchical col_indices length");
            size_t hier_col_idx = *reinterpret_cast<const size_t*>(data + offset);
            offset += sizeof(size_t);
            if (hier_col_idx > 0) {
                MMAP_CHECK_BOUNDS(hier_col_idx * sizeof(int), "Hierarchical col_indices data");
                offset += hier_col_idx * sizeof(int);
            }

            // Read number of children
            MMAP_CHECK_BOUNDS(sizeof(size_t), "Hierarchical num_children");
            size_t num_children = *reinterpret_cast<const size_t*>(data + offset);
            offset += sizeof(size_t);

            // Add this block first (so children can reference it)
            out_block_idx = mmap_blocks_.size();
            mmap_blocks_.push_back(std::move(info));

            // Recursively scan children
            for (size_t c = 0; c < num_children; ++c) {
                MMAP_CHECK_BOUNDS(sizeof(bool), "Hierarchical has_child flag");
                bool has_child = *reinterpret_cast<const bool*>(data + offset);
                offset += sizeof(bool);

                if (has_child) {
                    size_t child_idx;
                    if (!scanBlockForMmap(data, file_size, offset, child_idx)) {
                        return false;
                    }
                    mmap_blocks_[out_block_idx].child_indices.push_back(child_idx);
                }
            }
            return true;  // Already added
        }
        default:
            std::cerr << "  [mmap scan] unknown block type " << static_cast<int>(type)
                      << " at offset " << block_start_offset << std::endl;
            return false;
    }

    out_block_idx = mmap_blocks_.size();
    mmap_blocks_.push_back(std::move(info));
    #undef MMAP_CHECK_BOUNDS
    return true;
}

bool HMatrixViewFactor::readBinaryMmap(const std::string& filename) {
    // Clean up any existing mmap
    cleanupMmap();

    // Open file
    mmap_fd_ = open(filename.c_str(), O_RDONLY);
    if (mmap_fd_ < 0) {
        std::cerr << "Failed to open file for mmap: " << filename << std::endl;
        return false;
    }

    // Get file size
    struct stat st;
    if (fstat(mmap_fd_, &st) < 0) {
        std::cerr << "Failed to get file size: " << filename << std::endl;
        close(mmap_fd_);
        mmap_fd_ = -1;
        return false;
    }
    mmap_size_ = static_cast<size_t>(st.st_size);

    // Memory-map the file
    mmap_base_ = static_cast<char*>(mmap(nullptr, mmap_size_, PROT_READ, MAP_SHARED, mmap_fd_, 0));
    if (mmap_base_ == MAP_FAILED) {
        std::cerr << "Failed to mmap file: " << filename << std::endl;
        close(mmap_fd_);
        mmap_fd_ = -1;
        mmap_base_ = nullptr;
        return false;
    }

    // Advise kernel about access pattern
    madvise(mmap_base_, mmap_size_, MADV_RANDOM);  // Random access during matvec

    mmap_file_ = filename;

    // Parse header
    size_t offset = 0;
    const char* data = mmap_base_;

    // Read and verify magic/version
    if (offset + sizeof(uint32_t) * 2 > mmap_size_) {
        cleanupMmap();
        return false;
    }
    uint32_t magic = *reinterpret_cast<const uint32_t*>(data + offset);
    offset += sizeof(uint32_t);
    uint32_t version = *reinterpret_cast<const uint32_t*>(data + offset);
    offset += sizeof(uint32_t);

    if (magic != HMATRIX_MAGIC) {
        std::cerr << "Invalid H-matrix file format (bad magic)" << std::endl;
        cleanupMmap();
        return false;
    }
    if (version != HMATRIX_VERSION) {
        std::cerr << "Unsupported H-matrix version: " << version << std::endl;
        cleanupMmap();
        return false;
    }

    // Read dimensions
    if (offset + sizeof(size_t) * 3 > mmap_size_) {
        cleanupMmap();
        return false;
    }
    n_rows_ = *reinterpret_cast<const size_t*>(data + offset);
    offset += sizeof(size_t);
    n_cols_ = *reinterpret_cast<const size_t*>(data + offset);
    offset += sizeof(size_t);

    // Read number of root blocks
    size_t num_root_blocks = *reinterpret_cast<const size_t*>(data + offset);
    offset += sizeof(size_t);

    // Scan all blocks to build index
    mmap_blocks_.clear();
    mmap_blocks_.reserve(num_root_blocks * 10);  // Estimate
    mmap_root_indices_.clear();
    mmap_root_indices_.reserve(num_root_blocks);

    for (size_t b = 0; b < num_root_blocks; ++b) {
        if (offset + sizeof(bool) > mmap_size_) {
            cleanupMmap();
            return false;
        }
        bool has_block = *reinterpret_cast<const bool*>(data + offset);
        offset += sizeof(bool);

        if (has_block) {
            size_t block_idx;
            if (!scanBlockForMmap(data, mmap_size_, offset, block_idx)) {
                std::cerr << "Failed to scan block " << b << " in mmap" << std::endl;
                cleanupMmap();
                return false;
            }
            mmap_root_indices_.push_back(block_idx);
        }
    }

    // Read statistics from end of file
    // Position should now be at the statistics section
    if (offset + 6 * sizeof(size_t) <= mmap_size_) {
        num_zero_blocks_ = *reinterpret_cast<const size_t*>(data + offset);
        offset += sizeof(size_t);
        num_sparse_blocks_ = *reinterpret_cast<const size_t*>(data + offset);
        offset += sizeof(size_t);
        num_lowrank_blocks_ = *reinterpret_cast<const size_t*>(data + offset);
        offset += sizeof(size_t);
        num_dense_blocks_ = *reinterpret_cast<const size_t*>(data + offset);
        offset += sizeof(size_t);
        num_hierarchical_blocks_ = *reinterpret_cast<const size_t*>(data + offset);
        offset += sizeof(size_t);
        total_nnz_ = *reinterpret_cast<const size_t*>(data + offset);
    }

    mmap_loaded_ = true;

    std::cout << "H-matrix loaded via mmap from " << filename << std::endl;
    std::cout << "  File size: " << std::fixed << std::setprecision(2)
              << (mmap_size_ / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
    std::cout << "  Block index entries: " << mmap_blocks_.size() << std::endl;
    std::cout << "  Root blocks: " << mmap_root_indices_.size() << std::endl;
    std::cout << "  RAM usage: ~" << (mmap_blocks_.size() * sizeof(MmapBlockInfo) +
                                       mmap_root_indices_.size() * sizeof(size_t)) / (1024.0 * 1024.0)
              << " MB (index only)" << std::endl;
    printStatistics();

    return true;
}

// Perform matvec for a single mmap'd block
void HMatrixViewFactor::matvecMmapBlock(size_t block_idx, const double* x, double* y) const {
    if (block_idx >= mmap_blocks_.size()) return;

    const MmapBlockInfo& info = mmap_blocks_[block_idx];
    const char* data_ptr = mmap_base_ + info.file_offset;

    switch (info.block_type) {
        case 0: // Zero block - nothing to do
            break;

        case 1: { // Sparse block
            // Skip n_rows, n_cols (already in info)
            data_ptr += 2 * sizeof(int);

            // Read values
            size_t num_values = *reinterpret_cast<const size_t*>(data_ptr);
            data_ptr += sizeof(size_t);
            const double* values = reinterpret_cast<const double*>(data_ptr);
            data_ptr += num_values * sizeof(double);

            // Read col_indices
            size_t num_col_idx = *reinterpret_cast<const size_t*>(data_ptr);
            data_ptr += sizeof(size_t);
            const int* col_idx = reinterpret_cast<const int*>(data_ptr);
            data_ptr += num_col_idx * sizeof(int);

            // Read row_ptrs
            size_t num_row_ptrs = *reinterpret_cast<const size_t*>(data_ptr);
            data_ptr += sizeof(size_t);
            const int* row_ptrs = reinterpret_cast<const int*>(data_ptr);

            // Sparse matvec: y[row] += sum(values[k] * x[col])
            int n_rows = info.n_rows;
            for (int i = 0; i < n_rows; ++i) {
                int global_row = info.row_indices[i];
                double sum = 0.0;
                for (int k = row_ptrs[i]; k < row_ptrs[i + 1]; ++k) {
                    int local_col = col_idx[k];
                    int global_col = info.col_indices[local_col];
                    sum += values[k] * x[global_col];
                }
                #pragma omp atomic
                y[global_row] += sum;
            }
            break;
        }

        case 2: { // Low-rank block
            // Skip n_rows, n_cols, rank (already in info)
            data_ptr += 3 * sizeof(int);

            int n_rows = info.n_rows;
            int n_cols = info.n_cols;
            int rank = info.rank;

            if (rank == 0) break;

            // Read U dimensions and data (writeEigenMatrix uses int for rows/cols)
            int u_rows = *reinterpret_cast<const int*>(data_ptr);
            data_ptr += sizeof(int);
            int u_cols = *reinterpret_cast<const int*>(data_ptr);
            data_ptr += sizeof(int);
            const double* U_data = reinterpret_cast<const double*>(data_ptr);
            if (u_rows > 0 && u_cols > 0) {
                data_ptr += static_cast<size_t>(u_rows) * u_cols * sizeof(double);
            }

            // Read sigma (writeEigenVector uses int for size, NOT size_t!)
            int sigma_size = *reinterpret_cast<const int*>(data_ptr);
            data_ptr += sizeof(int);
            const double* sigma_data = reinterpret_cast<const double*>(data_ptr);
            if (sigma_size > 0) {
                data_ptr += static_cast<size_t>(sigma_size) * sizeof(double);
            }

            // Read V dimensions and data
            int v_rows = *reinterpret_cast<const int*>(data_ptr);
            data_ptr += sizeof(int);
            int v_cols = *reinterpret_cast<const int*>(data_ptr);
            data_ptr += sizeof(int);
            const double* V_data = reinterpret_cast<const double*>(data_ptr);

            // Low-rank matvec: y += U * diag(sigma) * V^T * x
            // Step 1: temp = V^T * x (gather from global x)
            std::vector<double> temp(rank, 0.0);
            for (int r = 0; r < rank; ++r) {
                double sum = 0.0;
                for (int j = 0; j < n_cols; ++j) {
                    int global_col = info.col_indices[j];
                    // V is stored column-major from Eigen
                    sum += V_data[j + r * v_rows] * x[global_col];
                }
                temp[r] = sigma_data[r] * sum;
            }

            // Step 2: y += U * temp (scatter to global y)
            for (int i = 0; i < n_rows; ++i) {
                int global_row = info.row_indices[i];
                double sum = 0.0;
                for (int r = 0; r < rank; ++r) {
                    sum += U_data[i + r * u_rows] * temp[r];
                }
                #pragma omp atomic
                y[global_row] += sum;
            }
            break;
        }

        case 3: { // Dense block
            data_ptr += 2 * sizeof(int);  // Skip n_rows, n_cols

            int d_rows = *reinterpret_cast<const int*>(data_ptr);
            data_ptr += sizeof(int);
            int d_cols = *reinterpret_cast<const int*>(data_ptr);
            data_ptr += sizeof(int);
            const double* dense_data = reinterpret_cast<const double*>(data_ptr);

            // Dense matvec
            for (int i = 0; i < d_rows; ++i) {
                int global_row = info.row_indices[i];
                double sum = 0.0;
                for (int j = 0; j < d_cols; ++j) {
                    int global_col = info.col_indices[j];
                    sum += dense_data[i + j * d_rows] * x[global_col];  // Column-major
                }
                #pragma omp atomic
                y[global_row] += sum;
            }
            break;
        }

        case 4: // Hierarchical block - recurse to children
            for (size_t child_idx : info.child_indices) {
                matvecMmapBlock(child_idx, x, y);
            }
            break;
    }
}

} // namespace radiation
