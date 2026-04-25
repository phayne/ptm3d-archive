#include "cluster_tree.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>

namespace radiation {

namespace {

inline int octantIndex(const std::array<double, 3>& point,
                       const std::array<double, 3>& center) {
    int o = 0;
    if (point[0] >= center[0]) o |= 1;
    if (point[1] >= center[1]) o |= 2;
    if (point[2] >= center[2]) o |= 4;
    return o;
}

inline std::array<double, 3> sub3(const std::array<double, 3>& a,
                                  const std::array<double, 3>& b) {
    return {{a[0] - b[0], a[1] - b[1], a[2] - b[2]}};
}

inline double dot3(const std::array<double, 3>& a,
                   const std::array<double, 3>& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

}  // namespace

void ClusterTree::buildRecursive(Node& node,
                                 const std::vector<std::array<double, 3>>& face_centers,
                                 std::vector<int>&& indices,
                                 int max_depth,
                                 int min_size,
                                 int current_depth) {
    node.depth = current_depth;

    if (indices.empty()) {
        node.is_leaf = true;
        return;
    }

    // AABB
    node.min_bound = face_centers[indices[0]];
    node.max_bound = node.min_bound;
    for (int idx : indices) {
        const auto& c = face_centers[idx];
        for (int k = 0; k < 3; ++k) {
            node.min_bound[k] = std::min(node.min_bound[k], c[k]);
            node.max_bound[k] = std::max(node.max_bound[k], c[k]);
        }
    }
    for (int k = 0; k < 3; ++k) {
        node.center[k] = 0.5 * (node.min_bound[k] + node.max_bound[k]);
    }

    if (current_depth >= max_depth ||
        static_cast<int>(indices.size()) <= min_size) {
        node.is_leaf = true;
        node.face_indices = std::move(indices);
        return;
    }

    std::array<std::vector<int>, 8> child_indices;
    for (int idx : indices) {
        child_indices[octantIndex(face_centers[idx], node.center)].push_back(idx);
    }

    int non_empty = 0;
    for (int o = 0; o < 8; ++o) {
        if (!child_indices[o].empty()) ++non_empty;
    }
    if (non_empty < 2) {
        node.is_leaf = true;
        node.face_indices = std::move(indices);
        return;
    }

    node.is_leaf = false;
    for (int o = 0; o < 8; ++o) {
        if (child_indices[o].empty()) continue;
        node.children[o] = std::make_unique<Node>();
        buildRecursive(*node.children[o], face_centers,
                       std::move(child_indices[o]),
                       max_depth, min_size, current_depth + 1);
    }
}

void ClusterTree::collectLeaves(
    Node& node,
    const std::vector<std::array<double, 3>>& face_centers,
    const std::vector<std::array<double, 3>>& face_normals) {
    if (node.is_leaf) {
        if (node.face_indices.empty()) return;
        int id = static_cast<int>(leaves_.size());
        node.leaf_id = id;
        Leaf leaf{};
        std::array<double, 3> c_sum{{0, 0, 0}};
        std::array<double, 3> n_sum{{0, 0, 0}};
        for (int idx : node.face_indices) {
            for (int k = 0; k < 3; ++k) {
                c_sum[k] += face_centers[idx][k];
                n_sum[k] += face_normals[idx][k];
            }
            leaf_id_[idx] = id;
        }
        double inv = 1.0 / static_cast<double>(node.face_indices.size());
        for (int k = 0; k < 3; ++k) {
            leaf.avg_center[k] = c_sum[k] * inv;
            leaf.avg_normal[k] = n_sum[k] * inv;  // intentionally not renormalized
        }
        // Normalize avg_normal so threshold comparison is a true cosine.
        double nrm = std::sqrt(dot3(leaf.avg_normal, leaf.avg_normal));
        if (nrm > 1e-20) {
            for (int k = 0; k < 3; ++k) leaf.avg_normal[k] /= nrm;
        }
        leaves_.push_back(leaf);
        return;
    }
    for (int o = 0; o < 8; ++o) {
        if (node.children[o]) {
            collectLeaves(*node.children[o], face_centers, face_normals);
        }
    }
}

void ClusterTree::build(const std::vector<std::array<double, 3>>& face_centers,
                        const std::vector<std::array<double, 3>>& face_normals,
                        int max_depth,
                        int min_size) {
    const size_t n = face_centers.size();
    leaf_id_.assign(n, -1);
    leaves_.clear();
    root_ = std::make_unique<Node>();

    std::vector<int> all(n);
    std::iota(all.begin(), all.end(), 0);
    buildRecursive(*root_, face_centers, std::move(all),
                   max_depth, min_size, /*current_depth=*/0);

    collectLeaves(*root_, face_centers, face_normals);
}

bool ClusterTree::canLeavesSeeEachOther(int a, int b) const {
    if (a < 0 || b < 0) return true;  // unknown -> conservative
    if (a == b) return true;
    if (a >= static_cast<int>(leaves_.size()) ||
        b >= static_cast<int>(leaves_.size())) {
        return true;
    }

    const Leaf& la = leaves_[a];
    const Leaf& lb = leaves_[b];

    auto r_vec = sub3(lb.avg_center, la.avg_center);
    double r2 = dot3(r_vec, r_vec);
    if (r2 < 1e-20) return true;  // overlapping centroids — be safe
    double inv_r = 1.0 / std::sqrt(r2);
    std::array<double, 3> r_hat = {{r_vec[0] * inv_r,
                                    r_vec[1] * inv_r,
                                    r_vec[2] * inv_r}};

    double dot_a = dot3(la.avg_normal, r_hat);   // a's avg normal vs direction a->b
    double dot_b = dot3(lb.avg_normal, r_hat);   // b's avg normal vs direction a->b

    // Conservative cull: only prune if BOTH clusters clearly back-face.
    // Threshold mirrors hmatrix_view_factor.cpp:1422 (CLUSTER_BACKFACE_THRESHOLD = -0.5).
    // - la's avg normal should point toward b (dot_a > 0); if strongly opposite (< -0.5) -> cull
    // - lb's avg normal should point toward a (dot_b < 0); if strongly same-direction (> +0.5) -> cull
    constexpr double kBackfaceThreshold = 0.5;
    if (dot_a < -kBackfaceThreshold) return false;
    if (dot_b >  kBackfaceThreshold) return false;
    return true;
}

size_t ClusterTree::memoryBytes() const {
    size_t bytes = sizeof(*this);
    bytes += leaves_.capacity() * sizeof(Leaf);
    bytes += leaf_id_.capacity() * sizeof(int);
    // Node tree memory is harder to tally precisely; rough lower bound.
    bytes += static_cast<size_t>(numLeaves()) * sizeof(Node);
    return bytes;
}

}  // namespace radiation
