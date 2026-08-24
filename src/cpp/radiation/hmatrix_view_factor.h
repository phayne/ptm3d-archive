#pragma once

#include <vector>
#include <array>
#include <memory>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <mutex>
#include <omp.h>
#include <variant>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "radiation_config.h"
#include "nearfield_quadrature.h"
#include "../mesh/mesh_utils.h"
#include "horizon_field.h"

// Forward declarations for Embree
typedef struct RTCDeviceTy* RTCDevice;
typedef struct RTCSceneTy* RTCScene;

namespace radiation {

// Forward declarations
class HMatrixViewFactor;
struct HMatrixBlock;

// SVD algorithm selection
enum class SVDMethod {
    Jacobi,      // Eigen::JacobiSVD - exact but O(m*n*min(m,n))
    Randomized   // Randomized SVD - approximate but O(m*n*k) where k is target rank
};

// Compression method selection
enum class CompressionMethod {
    SVD,         // Compute full block, then compress with SVD (original method)
    ACA          // Adaptive Cross Approximation - builds low-rank directly (much faster)
};

// Configuration for H-matrix compression
struct HMatrixConfig {
    double compression_tolerance = 1e-2;  // SVD truncation tolerance (relative to largest singular value)
    int max_tree_depth = 10;              // Maximum quadtree depth
    int min_block_size = 64;              // Minimum block size before treating as leaf
    bool use_sparse_svd = true;           // Use sparse SVD for compression
    int max_svd_rank = 50;                // Maximum rank for low-rank approximations
    bool verbose = false;                 // Print progress information
    SVDMethod svd_method = SVDMethod::Randomized;  // SVD algorithm to use
    int randomized_oversampling = 10;     // Oversampling parameter for randomized SVD (p)
    int randomized_power_iterations = 2;  // Power iterations for randomized SVD (q)

    // ACA configuration (Adaptive Cross Approximation)
    CompressionMethod compression_method = CompressionMethod::ACA;  // Default to ACA for speed
    double aca_tolerance = 1e-3;          // ACA relative tolerance for convergence
    int aca_max_rank = 50;                // Maximum rank before falling back to sparse
    int aca_min_block_size = 100;         // Minimum block size to use ACA (smaller uses SVD)

    // Hierarchical subdivision control
    int max_hierarchical_levels = 2;      // Maximum levels of hierarchical subdivision beyond start depth

    // Admissible-block subdivision (robustness fix for crater floor<->wall coupling).
    // When ACA on a large admissible block would be too expensive (side > aca_cap) or
    // ACA fails on a large block, bisect the FACE-INDEX LISTS and re-attempt ACA on
    // compact sub-clusters instead of zeroing the block. OFF reproduces the exact old
    // zero behavior for A/B verification. Env PTM_HMATRIX_SUBDIVIDE (1/0) overrides this.
    bool subdivide_large_admissible = true;
    int  subdiv_leaf_size = 500;          // leaf side; matches the old MAX_DIRECT_COMPUTE

    // Out-of-core (disk-backed) configuration for very large meshes
    bool use_out_of_core = false;         // Enable disk-backed storage during build
    std::string out_of_core_path = "";    // Path for temporary block storage
    size_t memory_limit_gb = 0;           // Auto-enable out-of-core if estimate exceeds this (0 = disabled)

    // Memory-mapped loading configuration
    bool use_mmap_loading = false;        // Load H-matrix via mmap instead of into RAM
    size_t mmap_threshold_faces = 500000; // Auto-enable mmap loading for meshes larger than this

    // Thread control for H-matrix building
    int hmatrix_threads = 0;              // 0 = auto, >0 = manual thread limit
};

// Quadtree node for spatial partitioning of face indices (2D - legacy)
struct QuadtreeNode {
    std::vector<int> face_indices;  // Indices of faces in this node
    std::array<double, 2> min_bound;  // 2D bounding box (x,y projection)
    std::array<double, 2> max_bound;
    std::array<double, 2> center;
    double diameter;  // Diameter of the bounding box

    std::unique_ptr<QuadtreeNode> children[4];  // NW, NE, SW, SE
    bool is_leaf = true;
    int depth = 0;

    QuadtreeNode() = default;

    // Build quadtree recursively
    void build(const std::vector<std::array<double, 3>>& face_centers,
               const std::vector<int>& indices,
               int max_depth, int min_size, int current_depth = 0);

    // Get all leaf nodes
    void getLeaves(std::vector<QuadtreeNode*>& leaves);

    // Get all nodes at a given depth (or all leaves if depth is too large)
    void getNodesAtDepth(int target_depth, std::vector<QuadtreeNode*>& nodes);
};

// Octree node for 3D spatial partitioning of face indices
// Better than quadtree for crater geometry where Z variation matters
struct OctreeNode {
    std::vector<int> face_indices;  // Indices of faces in this node
    std::array<double, 3> min_bound;  // 3D bounding box
    std::array<double, 3> max_bound;
    std::array<double, 3> center;    // Center of bounding box
    double diameter;  // Diameter of the bounding box (3D diagonal)

    std::unique_ptr<OctreeNode> children[8];  // 8 octants
    bool is_leaf = true;
    int depth = 0;

    OctreeNode() = default;

    // Build octree recursively
    void build(const std::vector<std::array<double, 3>>& face_centers,
               const std::vector<int>& indices,
               int max_depth, int min_size, int current_depth = 0);

    // Get all leaf nodes
    void getLeaves(std::vector<OctreeNode*>& leaves);

    // Get all nodes at a given depth (or all leaves if depth is too large)
    void getNodesAtDepth(int target_depth, std::vector<OctreeNode*>& nodes);

    // Get octant index for a point (0-7)
    int getOctant(const std::array<double, 3>& point) const {
        int octant = 0;
        if (point[0] >= center[0]) octant |= 1;
        if (point[1] >= center[1]) octant |= 2;
        if (point[2] >= center[2]) octant |= 4;
        return octant;
    }
};

// Block types for the hierarchical matrix
// 1. Zero block - no visible faces between clusters
// 2. Sparse CSR block - explicit sparse storage
// 3. Low-rank block - U * Sigma * V^T factorization
// 4. Dense block - full matrix storage
// 5. Hierarchical block - recursively subdivided

struct ZeroBlock {
    int n_rows = 0;
    int n_cols = 0;

    // No-op: zero block contributes nothing to matrix-vector product
    void matvec(const double* /*x*/, double* /*y*/) const { /* empty */ }
    size_t memoryBytes() const { return 0; }
};

struct SparseBlock {
    int n_rows = 0;
    int n_cols = 0;
    std::vector<double> values;
    std::vector<int> col_indices;
    std::vector<int> row_ptrs;

    void matvec(const double* x, double* y) const;
    size_t memoryBytes() const;
};

struct LowRankBlock {
    int n_rows = 0;
    int n_cols = 0;
    int rank = 0;

    // Store as U (n_rows x rank), sigma (rank), V (n_cols x rank)
    // so F ≈ U * diag(sigma) * V^T
    Eigen::MatrixXd U;
    Eigen::VectorXd sigma;
    Eigen::MatrixXd V;

    void matvec(const double* x, double* y) const;
    size_t memoryBytes() const;
};

struct DenseBlock {
    int n_rows = 0;
    int n_cols = 0;
    Eigen::MatrixXd data;

    void matvec(const double* x, double* y) const;
    size_t memoryBytes() const;
};

struct HierarchicalBlock {
    int n_rows = 0;
    int n_cols = 0;

    // Dynamic children storage - can have up to 16 children (4 row quadrants x 4 col quadrants)
    // Each child represents a (row_quadrant, col_quadrant) pair
    std::vector<std::unique_ptr<HMatrixBlock>> children;

    // Row and column index mappings
    std::vector<int> row_indices;  // Maps local row to global row
    std::vector<int> col_indices;  // Maps local col to global col

    void matvec(const double* x, double* y) const;
    size_t memoryBytes() const;
};

// Disk-backed block - stores reference to data on disk instead of in memory
// Used for out-of-core processing of very large meshes
struct DiskBlock {
    int n_rows = 0;
    int n_cols = 0;

    // Type of block stored on disk (0=zero, 1=sparse, 2=lowrank)
    int block_type = 0;

    // File offset and size for the block data
    int64_t file_offset = 0;
    int64_t data_size = 0;

    // For sparse blocks: number of non-zeros
    int nnz = 0;

    // For low-rank blocks: rank
    int rank = 0;

    // Pointer to the memory-mapped file (set by parent HMatrixViewFactor)
    const char* mmap_base = nullptr;

    // Full matvec with global arrays - reads indices from mmap file (memory-efficient)
    // This is the primary matvec method for DiskBlocks
    void matvec(const double* x, double* y) const;

    // Compatibility matvec for std::visit (delegates to primary method)
    // Note: row_indices/col_indices parameters are ignored - indices are read from disk
    void matvec(const double* x, double* y,
                const std::vector<int>& /*row_indices*/,
                const std::vector<int>& /*col_indices*/) const {
        matvec(x, y);  // Delegate to primary method that reads indices from disk
    }

    size_t memoryBytes() const { return sizeof(DiskBlock); }  // Only metadata in RAM
};

// Variant type for block storage
struct HMatrixBlock {
    std::variant<ZeroBlock, SparseBlock, LowRankBlock, DenseBlock, HierarchicalBlock, DiskBlock> data;

    // Row and column index mappings (global indices)
    std::vector<int> row_indices;
    std::vector<int> col_indices;

    int numRows() const;
    int numCols() const;
    void matvec(const double* x, double* y) const;
    size_t memoryBytes() const;
};

// Main hierarchical view factor matrix class
class HMatrixViewFactor {
public:
    HMatrixViewFactor() = default;

    // Build from mesh using Embree for visibility
    void build(const std::shared_ptr<ptm::Mesh>& mesh,
               RTCDevice device,
               RTCScene scene,
               const RadiationConfig& rad_config,
               const HMatrixConfig& hmat_config);

    // Matrix-vector product: y = F * x
    void matvec(const std::vector<double>& x, std::vector<double>& y) const;

    // Get matrix dimensions
    size_t numRows() const { return n_rows_; }
    size_t numCols() const { return n_cols_; }

    // Configure an EMPTY (nnz=0) n x n matrix with no blocks. Lets a tool emit a
    // correctly-dimensioned base .vf via writeBinary WITHOUT the multi-GB build:
    // it satisfies the VF-load gate (ptm_thermal_sim only rebuilds when the file
    // is MISSING or fails to load) and is overridden at run time by
    // PTM_HMATRIX_VF_REPLACE. All block/nnz stats are zeroed so writeBinary emits
    // a valid HMAT/v2 file that readBinary loads back with numRows()==n, nnz==0.
    void setEmptyDimensions(size_t n) {
        n_rows_ = n; n_cols_ = n;
        root_blocks_.clear();
        num_zero_blocks_ = num_sparse_blocks_ = num_lowrank_blocks_ =
            num_dense_blocks_ = num_hierarchical_blocks_ = total_nnz_ = 0;
    }

    // Memory usage
    size_t memoryBytes() const;
    double compressionRatio() const;  // Memory savings vs dense

    // Statistics
    void printStatistics() const;

    // How many pairs the ACA cross generators (computeViewFactorRow/Column)
    // evaluated on the most recent build(), and how many of those fell inside
    // this builder's near-field window. Process-wide, reset by build().
    //
    // These are exposure counters, not verdicts. They exist because an assertion
    // that the cross generators treat a near pair the same way the dense block
    // generator does is VACUOUS on a mesh where they never meet one, and this
    // project has already shipped one silently vacuous gate arm.
    static size_t crossGeneratorPairs();
    static size_t crossGeneratorNearPairs();
    // Closest approach the cross generators reached, as r / (4 d). Below 1 is
    // inside the near-field window; the value says how much margin there was.
    static double crossGeneratorMinROver4d();

    // The EXACT matrix this builder would produce with no hierarchical
    // decomposition and no low-rank approximation anywhere: one call to the
    // dense-block generator over every (i, j), expanded row-major into `out`
    // (n_faces^2 doubles). Same kernel, same near/far switch, same grazing
    // cull, same batched occlusion rays as a production dense block -- the ONLY
    // thing removed is the block structure. A difference between this and the
    // assembled matrix is therefore attributable to the decomposition and to
    // nothing else.
    //
    // Must be called after build(), which is what supplies the mesh, the scene
    // and the RadiationConfig. O(n^2) ray-traced entries and O(n^2) doubles, so
    // this is for gates and diagnostics on small meshes, not for production.
    //
    // NOT equivalent to building with min_block_size >= n_faces: that route
    // still passes the single block through tryCompressToLowRank(), which caps
    // the rank at max_svd_rank (default 50) and returns success even when the
    // requested tolerance was NOT met -- on a 12,800-face mesh that reference
    // came back with NEGATIVE view factors. Measured 2026-08-23; see
    // docs/design/vf-unification.md.
    void computeDenseReference(std::vector<double>& out);

    // Save/load compressed matrix
    bool writeBinary(const std::string& filename) const;
    bool readBinary(const std::string& filename);

    // Memory-mapped loading (for very large matrices)
    // When enabled, blocks are read on-demand from mmap during matvec
    bool readBinaryMmap(const std::string& filename);
    bool isMemoryMapped() const { return mmap_loaded_; }

    // Destructor needed to clean up mmap resources
    ~HMatrixViewFactor();

    // Disable copy (mmap resources can't be copied)
    HMatrixViewFactor(const HMatrixViewFactor&) = delete;
    HMatrixViewFactor& operator=(const HMatrixViewFactor&) = delete;

    // Allow move
    HMatrixViewFactor(HMatrixViewFactor&& other) noexcept;
    HMatrixViewFactor& operator=(HMatrixViewFactor&& other) noexcept;

private:
    // Build a single block of the H-matrix (quadtree version - legacy)
    std::unique_ptr<HMatrixBlock> buildBlock(
        QuadtreeNode* row_node,
        QuadtreeNode* col_node,
        int depth);

    // Build a single block of the H-matrix (octree version - preferred for 3D)
    std::unique_ptr<HMatrixBlock> buildBlock(
        OctreeNode* row_node,
        OctreeNode* col_node,
        int depth);

    // Bounded recursive subdivision of a LARGE admissible block. Bisects the
    // face-index lists directly (NOT octree children, which may be leaves),
    // re-attempts ACA on compact sub-clusters, and computes small leaves as exact
    // sparse. Returns a HierarchicalBlock wrapping the sub-blocks, a single leaf
    // block, or (last resort at the depth cap) a ZeroBlock. Replaces the old
    // "zero the large admissible block" fallback so crater floor<->wall coupling
    // survives above ~100k faces. See buildBlock() admissible path.
    std::unique_ptr<HMatrixBlock> buildAdmissibleSubdivided(
        const std::vector<int>& row_faces,
        const std::vector<int>& col_faces,
        int subdiv_depth,
        bool try_aca = true);

    // Compute sparse view factor submatrix for given row/col indices
    void computeViewFactorBlock(
        const std::vector<int>& row_faces,
        const std::vector<int>& col_faces,
        SparseBlock& sparse);

    // Try to compress a sparse block to low-rank
    bool tryCompressToLowRank(
        const SparseBlock& sparse,
        LowRankBlock& lowrank,
        double tolerance);

    // =========== ACA (Adaptive Cross Approximation) methods ===========

    // Compute a single row of the view factor matrix for ACA
    // result[j] = F[local_row, j] for j in col_faces
    void computeViewFactorRow(
        int local_row,
        const std::vector<int>& row_faces,
        const std::vector<int>& col_faces,
        std::vector<double>& result) const;

    // Compute a single column of the view factor matrix for ACA
    // result[i] = F[i, local_col] for i in row_faces
    void computeViewFactorColumn(
        int local_col,
        const std::vector<int>& row_faces,
        const std::vector<int>& col_faces,
        std::vector<double>& result) const;

    // ACA compression: builds low-rank directly without computing full block
    // Returns true if compression successful, false if should fall back to sparse
    bool tryCompressToLowRankACA(
        const std::vector<int>& row_faces,
        const std::vector<int>& col_faces,
        LowRankBlock& lowrank,
        double tolerance);

    // Check admissibility condition (are clusters well-separated?)
    // Quadtree version (2D distance)
    bool isAdmissible(QuadtreeNode* row_node, QuadtreeNode* col_node) const;
    // Octree version (3D distance - more accurate for crater geometry)
    bool isAdmissible(OctreeNode* row_node, OctreeNode* col_node) const;

    // Cluster-level visibility culling (conservative check using average normals)
    bool canClustersSeeEachOther(QuadtreeNode* row_node, QuadtreeNode* col_node) const;
    bool canClustersSeeEachOther(OctreeNode* row_node, OctreeNode* col_node) const;

    // Compute view factor between two faces
    double computeViewFactor(int face_i, int face_j) const;

    // Near-field-accurate view factor via area-to-area numerical quadrature.
    // The M1 centroid estimator A_j*cos_i*cos_j/(pi*r^2) is a far-field
    // approximation (valid only for r >> face size) and overshoots for close
    // or adjacent pairs, which is what makes Sum_j F_ij exceed 1 and the
    // radiosity coupling diverge. This double-integrates over both triangles,
    // subdividing more finely as the pair gets closer.
    //
    // Since 2026-08-23 it calls the SHARED estimator in nearfield_quadrature.h,
    // the same one the sparse builder uses. The old claim here -- "physically
    // bounded (Sum_j F_ij <= 1) with no artificial cap" -- was FALSE: the
    // quadrature diverges on a closing concave crease (F_ij = 4.59 at a 10
    // degree dihedral, 16.75 at 5, even at the adaptive ceiling). The bound is
    // now enforced by the shared kernel's clamp and reported by its tripwire.
    double computeViewFactorRefined(int face_i, int face_j) const;

    // Quadrature order for computeViewFactorRefined on this pair:
    // clamp(max(rad_config_.nearfield_quadrature_order, ceil(4 d / r)), 1,
    // kMaxNearFieldQuadOrder), d = sqrt(A_i) + sqrt(A_j). The configured key is
    // a FLOOR here, not the value -- see the measurement table at the
    // definition for why pinning it to the sparse path's fixed order was
    // rejected.
    int refinedQuadratureOrder(int face_i, int face_j) const;

    // The near/far switch, for every entry generator in this builder. `r` is
    // the centroid separation, already computed by every caller.
    bool isNearFieldPair(int face_i, int face_j, double r) const;

    // Visibility check
    bool isVisible(const std::array<double, 3>& p1,
                   const std::array<double, 3>& p2) const;

    // Vector utilities
    static std::array<double, 3> subtract(const std::array<double, 3>& a,
                                          const std::array<double, 3>& b);
    static double dot(const std::array<double, 3>& a,
                      const std::array<double, 3>& b);
    static double length(const std::array<double, 3>& v);
    static std::array<double, 3> cross(const std::array<double, 3>& a,
                                       const std::array<double, 3>& b);

    // Data members
    size_t n_rows_ = 0;
    size_t n_cols_ = 0;

    // Root blocks of the H-matrix (one per quadrant pair)
    std::vector<std::unique_ptr<HMatrixBlock>> root_blocks_;

    // Precomputed face data
    std::vector<std::array<double, 3>> face_centers_;
    std::vector<std::array<double, 3>> face_normals_;
    std::vector<double> face_areas_;

    // Heightfield-horizon visibility (robust grazing-tangent occlusion; recovers
    // crater floor<->rim coupling that the Embree thin-ray over-occludes on rough
    // fine meshes). Built in build() when env PTM_HMATRIX_HORIZON_VIS=1; used in
    // computeViewFactorRow/Column/Block instead of the Embree ray.
    HorizonField horizon_;
    bool use_horizon_vis_ = false;

    // Smoothed (local-mean/macroscopic) face normals for the view-factor cosine
    // terms. The raw per-facet normals on fine rough meshes tilt ~10deg off the
    // macroscopic surface and decouple grazing floor<->rim pairs (their cosine
    // lobes miss the rim); the macroscopic normal gives the resolution-independent
    // VF (validated by bin/vf_hemisphere_mc). Built in build() when env
    // PTM_HMATRIX_SMOOTH_NORMALS=1 (radius PTM_HMATRIX_SMOOTH_RADIUS, default ~300 m).
    std::vector<std::array<double, 3>> smoothed_normals_;
    bool use_smoothed_normals_ = false;
    void computeSmoothedNormals(double radius);
    // Accessor used by the view-factor cosine terms (NOT the cluster cull, which
    // already averages over the cluster).
    inline const std::array<double, 3>& vfNormal(int i) const {
        return use_smoothed_normals_ ? smoothed_normals_[i] : face_normals_[i];
    }

    // Spatial partitioning trees
    std::unique_ptr<QuadtreeNode> quadtree_root_;  // 2D quadtree (legacy)
    std::unique_ptr<OctreeNode> octree_root_;      // 3D octree (preferred for craters)
    bool use_octree_ = true;  // Use 3D octree instead of 2D quadtree
    int start_depth_ = 0;     // Depth at which block building started (for hierarchical limit)

    // Configuration
    HMatrixConfig config_;
    RadiationConfig rad_config_;

    // Mesh and Embree scene (for visibility)
    std::shared_ptr<ptm::Mesh> mesh_;
    RTCDevice device_ = nullptr;
    RTCScene scene_ = nullptr;

    // Statistics
    mutable size_t num_zero_blocks_ = 0;
    mutable size_t num_sparse_blocks_ = 0;
    mutable size_t num_lowrank_blocks_ = 0;
    mutable size_t num_dense_blocks_ = 0;
    mutable size_t num_hierarchical_blocks_ = 0;
    mutable size_t num_disk_blocks_ = 0;
    mutable size_t total_nnz_ = 0;

    // Out-of-core support
    bool out_of_core_enabled_ = false;
    std::string out_of_core_file_;
    int out_of_core_fd_ = -1;           // File descriptor for writing
    char* out_of_core_mmap_ = nullptr;  // Memory-mapped region for reading
    size_t out_of_core_size_ = 0;       // Size of mapped region
    mutable std::mutex out_of_core_mutex_;  // Protects file writes

    // Write block data to disk, return DiskBlock descriptor
    DiskBlock writeBlockToDisk(const SparseBlock& sparse,
                               const std::vector<int>& row_indices,
                               const std::vector<int>& col_indices);
    DiskBlock writeBlockToDisk(const LowRankBlock& lowrank,
                               const std::vector<int>& row_indices,
                               const std::vector<int>& col_indices);

    // Initialize out-of-core storage
    bool initOutOfCore(const std::string& path);
    void finalizeOutOfCore();  // Flush writes and setup mmap for reading
    void cleanupOutOfCore();   // Close files and unmap

    // Memory-mapped loading support (for reading saved H-matrix files)
    bool mmap_loaded_ = false;            // True if loaded via mmap
    std::string mmap_file_;               // Path to mmap'd file
    int mmap_fd_ = -1;                    // File descriptor
    char* mmap_base_ = nullptr;           // Base pointer to mmap'd region
    size_t mmap_size_ = 0;                // Size of mmap'd region

    // Block offset table for mmap loading (built during readBinaryMmap)
    // Each entry: [file_offset, block_type, n_rows, n_cols, extra_info...]
    struct MmapBlockInfo {
        int64_t file_offset = 0;          // Offset in file where block data starts
        int block_type = 0;               // 0=zero, 1=sparse, 2=lowrank, 3=dense, 4=hierarchical
        int n_rows = 0;
        int n_cols = 0;
        int nnz = 0;                      // For sparse blocks
        int rank = 0;                     // For low-rank blocks
        std::vector<int> row_indices;     // Global row indices
        std::vector<int> col_indices;     // Global column indices
        std::vector<size_t> child_indices; // For hierarchical blocks: indices into mmap_blocks_
    };
    std::vector<MmapBlockInfo> mmap_blocks_;  // Flat list of all blocks
    std::vector<size_t> mmap_root_indices_;   // Indices of root blocks in mmap_blocks_

    // Helper to perform matvec using mmap'd data
    void matvecMmapBlock(size_t block_idx, const double* x, double* y) const;

    // Helper to scan a block during mmap loading
    bool scanBlockForMmap(const char* data, size_t file_size, size_t& offset, size_t& out_block_idx);

    // Cleanup mmap resources
    void cleanupMmap();
};

} // namespace radiation
