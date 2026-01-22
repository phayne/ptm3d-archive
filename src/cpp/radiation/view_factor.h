#pragma once

#include <vector>
#include <array>
#include <memory>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <omp.h>

#include "radiation_config.h"
#include "../mesh/mesh_utils.h"

// Forward declarations for Embree
typedef struct RTCDeviceTy* RTCDevice;
typedef struct RTCSceneTy* RTCScene;

namespace radiation {

// Sparse Compressed Sparse Row (CSR) storage for view factor matrix
// Stores F_ij where i is the row (receiving face) and j is the column (emitting face)
class ViewFactorMatrix {
public:
    ViewFactorMatrix() = default;

    // Initialize for n_faces (reserve estimated capacity)
    void initialize(size_t n_faces, size_t estimated_nnz_per_row = 100);

    // Build from COO (coordinate) format triplets
    void buildFromTriplets(size_t n_faces,
                           const std::vector<int>& row_indices,
                           const std::vector<int>& col_indices,
                           const std::vector<double>& values);

    // Access elements
    double get(int row, int col) const;

    // Get all non-zero entries in a row
    void getRow(int row, std::vector<int>& cols, std::vector<double>& vals) const;

    // Get number of non-zeros in a row
    int getRowNNZ(int row) const;

    // Matrix-vector product: y = F * x (where F is this matrix)
    void matvec(const std::vector<double>& x, std::vector<double>& y) const;

    // Transposed matrix-vector product: y = F^T * x
    void matvecTranspose(const std::vector<double>& x, std::vector<double>& y) const;

    // Get dimensions
    size_t numRows() const { return n_rows_; }
    size_t numCols() const { return n_cols_; }
    size_t numNonZeros() const { return values_.size(); }

    // Get raw data for advanced operations
    const std::vector<double>& values() const { return values_; }
    const std::vector<int>& colIndices() const { return col_indices_; }
    const std::vector<int>& rowPointers() const { return row_ptrs_; }

    // Compute row sums (useful for checking sum rule: sum_j F_ij + F_sky = 1)
    std::vector<double> rowSums() const;

    // Clear all data
    void clear();

    // Memory usage in bytes
    size_t memoryUsage() const;

    // Write to binary file
    bool writeBinary(const std::string& filename) const;

    // Read from binary file
    bool readBinary(const std::string& filename);

private:
    size_t n_rows_ = 0;
    size_t n_cols_ = 0;
    std::vector<double> values_;      // Non-zero values
    std::vector<int> col_indices_;    // Column indices
    std::vector<int> row_ptrs_;       // Row pointers (size = n_rows + 1)
};

// View factor calculator using Embree for visibility
class ViewFactorCalculator {
public:
    // Constructor takes mesh and Embree scene (scene must already be committed)
    ViewFactorCalculator(const std::shared_ptr<ptm::Mesh>& mesh,
                         RTCDevice device,
                         RTCScene scene,
                         const RadiationConfig& config);

    ~ViewFactorCalculator();

    // Compute view factors for all faces in the mesh
    // Returns sparse matrix of F_ij values
    ViewFactorMatrix computeAll();

    // Compute view factors only for specified faces (subset computation)
    // face_indices: global indices of faces to compute view factors for
    // Returns matrix where rows correspond to face_indices order
    ViewFactorMatrix computeSubset(const std::vector<int>& face_indices);

    // Compute single view factor between two faces (for testing/debugging)
    double computePair(int face_i, int face_j);

    // Compute sky view factor for a face (1 - sum of terrain view factors)
    double computeSkyViewFactor(int face_idx);

    // Get precomputed face data
    const std::vector<std::array<double, 3>>& getFaceCenters() const { return face_centers_; }
    const std::vector<std::array<double, 3>>& getFaceNormals() const { return face_normals_; }
    const std::vector<double>& getFaceAreas() const { return face_areas_; }

private:
    // View factor methods
    double computeCentroid(int face_i, int face_j);
    double computeSubdivision(int face_i, int face_j, int level);
    double computeAdaptive(int face_i, int face_j);

    // Visibility check using Embree
    bool isVisible(const std::array<double, 3>& p1, const std::array<double, 3>& p2);

    // Batch visibility check (more efficient)
    std::vector<bool> checkVisibilityBatch(
        const std::vector<std::array<double, 3>>& origins,
        const std::vector<std::array<double, 3>>& targets);

    // Find potential neighbors within search radius
    std::vector<int> findNeighbors(int face_idx, double radius);

    // Precompute face geometry
    void precomputeFaceData();

    // Compute face area
    double computeFaceArea(int face_idx);

    // Vector utilities
    static std::array<double, 3> subtract(const std::array<double, 3>& a,
                                          const std::array<double, 3>& b);
    static std::array<double, 3> cross(const std::array<double, 3>& a,
                                       const std::array<double, 3>& b);
    static double dot(const std::array<double, 3>& a,
                      const std::array<double, 3>& b);
    static double length(const std::array<double, 3>& v);
    static std::array<double, 3> normalize(const std::array<double, 3>& v);

    // Subdivision sampling points on a triangle
    void getSubdivisionPoints(int face_idx, int level,
                              std::vector<std::array<double, 3>>& points,
                              std::vector<double>& weights);

    // Data members
    std::shared_ptr<ptm::Mesh> mesh_;
    RTCDevice device_;
    RTCScene scene_;
    RadiationConfig config_;

    // Precomputed face data
    std::vector<std::array<double, 3>> face_centers_;
    std::vector<std::array<double, 3>> face_normals_;
    std::vector<double> face_areas_;
    bool face_data_computed_ = false;

    // Spatial acceleration (simple grid for now)
    struct SpatialGrid {
        std::vector<std::vector<int>> cells;
        std::array<double, 3> min_bound;
        std::array<double, 3> max_bound;
        std::array<double, 3> cell_size;
        std::array<int, 3> grid_dims;

        void build(const std::vector<std::array<double, 3>>& centers, double cell_size_hint);
        std::vector<int> query(const std::array<double, 3>& center, double radius) const;
        int getCellIndex(const std::array<double, 3>& point) const;
    };
    SpatialGrid spatial_grid_;
    bool spatial_grid_built_ = false;
};

} // namespace radiation
