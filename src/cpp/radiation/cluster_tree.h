#pragma once

#include <array>
#include <memory>
#include <vector>

namespace radiation {

// Lightweight centroid octree + conservative cluster-pair visibility cull.
// Used by ViewFactorCalculator to pre-prune candidate neighbors before
// per-face ranking in the sparse view-factor path.
//
// Ported/trimmed from hmatrix_view_factor.cpp's OctreeNode + canClustersSeeEachOther;
// this version precomputes (avg_center, avg_normal) per leaf and maps
// every face to a leaf id so the per-candidate check is O(1) lookup + O(1) test.
class ClusterTree {
public:
    ClusterTree() = default;

    // Build from face centers + normals. max_depth and min_size control leaf granularity
    // (defaults mirror the H-matrix config). face_ids are implicit 0..n-1.
    void build(const std::vector<std::array<double, 3>>& face_centers,
               const std::vector<std::array<double, 3>>& face_normals,
               int max_depth = 10,
               int min_size = 64);

    // Leaf id for a face (-1 if not built)
    int leafId(int face_idx) const {
        return (face_idx >= 0 && static_cast<size_t>(face_idx) < leaf_id_.size())
                   ? leaf_id_[face_idx]
                   : -1;
    }

    int numLeaves() const { return static_cast<int>(leaves_.size()); }

    // Conservative cluster-pair cull: returns false only if the two leaves
    // CLEARLY cannot see each other (both avg normals strongly back-facing).
    // Matches the H-matrix threshold (-0.5). Never wrongly prunes a visible pair
    // in the expected crater geometry; a false-positive (kept candidate) is benign.
    bool canLeavesSeeEachOther(int leaf_a, int leaf_b) const;

    // Memory footprint (rough)
    size_t memoryBytes() const;

private:
    struct Node {
        std::array<double, 3> min_bound{{0, 0, 0}};
        std::array<double, 3> max_bound{{0, 0, 0}};
        std::array<double, 3> center{{0, 0, 0}};
        std::unique_ptr<Node> children[8];
        std::vector<int> face_indices;  // populated on leaves
        int leaf_id = -1;               // -1 on internal nodes
        bool is_leaf = true;
        int depth = 0;
    };

    struct Leaf {
        std::array<double, 3> avg_center{{0, 0, 0}};
        std::array<double, 3> avg_normal{{0, 0, 0}};  // NOT re-normalized; magnitude tracks consistency
    };

    void buildRecursive(Node& node,
                        const std::vector<std::array<double, 3>>& face_centers,
                        std::vector<int>&& indices,
                        int max_depth,
                        int min_size,
                        int current_depth);

    void collectLeaves(Node& node,
                       const std::vector<std::array<double, 3>>& face_centers,
                       const std::vector<std::array<double, 3>>& face_normals);

    std::unique_ptr<Node> root_;
    std::vector<Leaf> leaves_;   // leaves_[leaf_id]
    std::vector<int> leaf_id_;   // leaf_id_[face_idx] -> index into leaves_
};

}  // namespace radiation
