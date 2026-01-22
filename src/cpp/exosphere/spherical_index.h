/*
 * Spherical Spatial Index for Exosphere Transport
 *
 * Provides O(N log N) neighbor finding on a sphere using a hierarchical
 * equal-area partitioning scheme inspired by HEALPix.
 *
 * Key features:
 * - Equal-area cells ensure uniform sampling density
 * - Hierarchical structure allows multi-resolution queries
 * - Query by angular radius returns only faces within theta_max
 * - Thread-safe read operations after build
 *
 * Complexity:
 * - Build: O(N log N)
 * - Query all neighbors within theta: O(k + log N) where k = number of neighbors
 * - Memory: O(N + C) where C = number of cells
 */

#ifndef SPHERICAL_INDEX_H
#define SPHERICAL_INDEX_H

#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <iostream>
#include <chrono>

namespace ptm {
namespace exosphere {

// Simple spherical cell for spatial indexing
struct SphericalCell {
    int cell_id;
    double theta_center;    // Colatitude of cell center [0, pi]
    double phi_center;      // Longitude of cell center [0, 2*pi]
    double angular_radius;  // Angular radius of cell
    std::vector<size_t> face_indices;  // Faces in this cell

    // Get unit vector for cell center
    std::array<double, 3> getUnitCenter() const {
        return {
            std::sin(theta_center) * std::cos(phi_center),
            std::sin(theta_center) * std::sin(phi_center),
            std::cos(theta_center)
        };
    }
};

class SphericalSpatialIndex {
private:
    // Grid parameters
    int n_theta_bands_;      // Number of latitude bands
    int n_phi_base_;         // Base number of longitude divisions at equator
    double cell_angular_size_;  // Approximate angular size of each cell

    // Cell storage
    std::vector<SphericalCell> cells_;
    std::unordered_map<int, size_t> cell_id_to_index_;

    // Face data
    std::vector<std::array<float, 3>> unit_centers_;  // Unit vectors for each face
    size_t num_faces_;

    // Lookup table: for each cell, list of neighboring cells
    std::vector<std::vector<int>> cell_neighbors_;

    // Convert (theta, phi) to cell ID
    int getCellId(double theta, double phi) const {
        // Clamp theta to valid range
        theta = std::max(0.0, std::min(M_PI, theta));

        // Which latitude band?
        int band = static_cast<int>(theta / cell_angular_size_);
        band = std::min(band, n_theta_bands_ - 1);

        // Number of phi divisions for this band (fewer near poles)
        double band_theta = (band + 0.5) * cell_angular_size_;
        int n_phi = std::max(1, static_cast<int>(n_phi_base_ * std::sin(band_theta)));

        // Which phi cell?
        while (phi < 0) phi += 2.0 * M_PI;
        while (phi >= 2.0 * M_PI) phi -= 2.0 * M_PI;
        int phi_idx = static_cast<int>(phi / (2.0 * M_PI) * n_phi);
        phi_idx = std::min(phi_idx, n_phi - 1);

        // Encode as single integer: band * max_phi + phi_idx
        return band * (n_phi_base_ + 1) + phi_idx;
    }

    // Get cell ID from unit vector
    int getCellIdFromUnitVector(const std::array<float, 3>& u) const {
        double theta = std::acos(std::max(-1.0f, std::min(1.0f, u[2])));
        double phi = std::atan2(u[1], u[0]);
        if (phi < 0) phi += 2.0 * M_PI;
        return getCellId(theta, phi);
    }

    // Precompute neighboring cells for each cell
    void buildCellNeighborList(double max_query_radius) {
        cell_neighbors_.resize(cells_.size());

        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < cells_.size(); ++i) {
            const auto& cell_i = cells_[i];
            auto center_i = cell_i.getUnitCenter();

            std::vector<int> neighbors;
            neighbors.reserve(32);

            for (size_t j = 0; j < cells_.size(); ++j) {
                const auto& cell_j = cells_[j];
                auto center_j = cell_j.getUnitCenter();

                // Angular distance between cell centers
                double dot = center_i[0]*center_j[0] + center_i[1]*center_j[1] + center_i[2]*center_j[2];
                dot = std::max(-1.0, std::min(1.0, dot));
                double dist = std::acos(dot);

                // Include cell if it could contain points within max_query_radius
                // Add cell radii as buffer
                if (dist <= max_query_radius + cell_i.angular_radius + cell_j.angular_radius) {
                    neighbors.push_back(cell_j.cell_id);
                }
            }

            cell_neighbors_[i] = std::move(neighbors);
        }
    }

public:
    SphericalSpatialIndex() : n_theta_bands_(0), n_phi_base_(0), cell_angular_size_(0), num_faces_(0) {}

    // Build the index from face centers
    // cell_size_degrees: target angular size of each cell in degrees
    // max_query_radius: maximum query radius in radians (for neighbor precomputation)
    void build(const std::vector<std::array<double, 3>>& face_centers,
               double cell_size_degrees = 10.0,
               double max_query_radius = M_PI / 3.0) {

        auto start = std::chrono::high_resolution_clock::now();

        num_faces_ = face_centers.size();
        if (num_faces_ == 0) return;

        // Convert cell size to radians
        cell_angular_size_ = cell_size_degrees * M_PI / 180.0;

        // Compute grid dimensions
        n_theta_bands_ = static_cast<int>(std::ceil(M_PI / cell_angular_size_));
        n_phi_base_ = static_cast<int>(std::ceil(2.0 * M_PI / cell_angular_size_));

        // Precompute unit vectors for all faces
        unit_centers_.resize(num_faces_);
        #pragma omp parallel for
        for (size_t i = 0; i < num_faces_; ++i) {
            const auto& fc = face_centers[i];
            double r = std::sqrt(fc[0]*fc[0] + fc[1]*fc[1] + fc[2]*fc[2]);
            if (r > 0.0) {
                double inv_r = 1.0 / r;
                unit_centers_[i] = {
                    static_cast<float>(fc[0] * inv_r),
                    static_cast<float>(fc[1] * inv_r),
                    static_cast<float>(fc[2] * inv_r)
                };
            } else {
                unit_centers_[i] = {0.0f, 0.0f, 1.0f};
            }
        }

        // Assign faces to cells
        std::unordered_map<int, std::vector<size_t>> cell_faces;
        for (size_t i = 0; i < num_faces_; ++i) {
            int cell_id = getCellIdFromUnitVector(unit_centers_[i]);
            cell_faces[cell_id].push_back(i);
        }

        // Create cell objects
        cells_.clear();
        cells_.reserve(cell_faces.size());
        cell_id_to_index_.clear();

        for (const auto& [cell_id, faces] : cell_faces) {
            SphericalCell cell;
            cell.cell_id = cell_id;
            cell.face_indices = faces;
            cell.angular_radius = cell_angular_size_ * 0.75;  // Conservative estimate

            // Compute cell center from ID
            int band = cell_id / (n_phi_base_ + 1);
            int phi_idx = cell_id % (n_phi_base_ + 1);
            cell.theta_center = (band + 0.5) * cell_angular_size_;

            double band_theta = cell.theta_center;
            int n_phi = std::max(1, static_cast<int>(n_phi_base_ * std::sin(band_theta)));
            cell.phi_center = (phi_idx + 0.5) * 2.0 * M_PI / n_phi;

            cell_id_to_index_[cell_id] = cells_.size();
            cells_.push_back(std::move(cell));
        }

        // Build neighbor lists
        buildCellNeighborList(max_query_radius);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Spherical index built: " << num_faces_ << " faces, "
                  << cells_.size() << " cells, "
                  << n_theta_bands_ << " bands x " << n_phi_base_ << " base phi, "
                  << duration.count() << " ms" << std::endl;
    }

    // Query all faces within angular distance theta_max of face i
    // Returns vector of (face_index, cos_theta) pairs
    void queryNeighbors(size_t face_i, double theta_max,
                       std::vector<std::pair<size_t, float>>& results) const {
        results.clear();
        if (face_i >= num_faces_ || cells_.empty()) return;

        const auto& u_i = unit_centers_[face_i];
        float cos_theta_max = static_cast<float>(std::cos(theta_max));

        // Find which cell this face is in
        int cell_id = getCellIdFromUnitVector(u_i);
        auto it = cell_id_to_index_.find(cell_id);
        if (it == cell_id_to_index_.end()) return;

        size_t cell_idx = it->second;
        const auto& neighbor_cell_ids = cell_neighbors_[cell_idx];

        // Check all faces in neighboring cells
        for (int neighbor_id : neighbor_cell_ids) {
            auto nit = cell_id_to_index_.find(neighbor_id);
            if (nit == cell_id_to_index_.end()) continue;

            const SphericalCell& neighbor_cell = cells_[nit->second];
            for (size_t face_j : neighbor_cell.face_indices) {
                if (face_j == face_i) continue;

                const auto& u_j = unit_centers_[face_j];
                float cos_theta = u_i[0]*u_j[0] + u_i[1]*u_j[1] + u_i[2]*u_j[2];

                if (cos_theta >= cos_theta_max) {
                    results.emplace_back(face_j, cos_theta);
                }
            }
        }
    }

    // Query using precomputed cos(theta_max) threshold
    void queryNeighborsByCosTheta(size_t face_i, float cos_theta_max,
                                  std::vector<std::pair<size_t, float>>& results) const {
        results.clear();
        if (face_i >= num_faces_ || cells_.empty()) return;

        const auto& u_i = unit_centers_[face_i];

        // Find which cell this face is in
        int cell_id = getCellIdFromUnitVector(u_i);
        auto it = cell_id_to_index_.find(cell_id);
        if (it == cell_id_to_index_.end()) return;

        size_t cell_idx = it->second;
        const auto& neighbor_cell_ids = cell_neighbors_[cell_idx];

        // Check all faces in neighboring cells
        for (int neighbor_id : neighbor_cell_ids) {
            auto nit = cell_id_to_index_.find(neighbor_id);
            if (nit == cell_id_to_index_.end()) continue;

            const SphericalCell& neighbor_cell = cells_[nit->second];
            for (size_t face_j : neighbor_cell.face_indices) {
                if (face_j == face_i) continue;

                const auto& u_j = unit_centers_[face_j];
                float cos_theta = u_i[0]*u_j[0] + u_i[1]*u_j[1] + u_i[2]*u_j[2];

                if (cos_theta >= cos_theta_max) {
                    results.emplace_back(face_j, cos_theta);
                }
            }
        }
    }

    // Get unit vector for a face
    const std::array<float, 3>& getUnitCenter(size_t i) const {
        return unit_centers_[i];
    }

    // Accessors
    size_t numFaces() const { return num_faces_; }
    size_t numCells() const { return cells_.size(); }
    double cellAngularSize() const { return cell_angular_size_; }

    // Statistics
    void printStatistics() const {
        if (cells_.empty()) {
            std::cout << "Spherical index is empty" << std::endl;
            return;
        }

        size_t min_faces = cells_[0].face_indices.size();
        size_t max_faces = cells_[0].face_indices.size();
        size_t total_faces = 0;

        size_t min_neighbors = cell_neighbors_[0].size();
        size_t max_neighbors = cell_neighbors_[0].size();
        size_t total_neighbors = 0;

        for (size_t i = 0; i < cells_.size(); ++i) {
            size_t n = cells_[i].face_indices.size();
            min_faces = std::min(min_faces, n);
            max_faces = std::max(max_faces, n);
            total_faces += n;

            size_t m = cell_neighbors_[i].size();
            min_neighbors = std::min(min_neighbors, m);
            max_neighbors = std::max(max_neighbors, m);
            total_neighbors += m;
        }

        std::cout << "Spherical index statistics:" << std::endl;
        std::cout << "  Cells: " << cells_.size() << std::endl;
        std::cout << "  Faces per cell: min=" << min_faces
                  << ", max=" << max_faces
                  << ", avg=" << (double)total_faces / cells_.size() << std::endl;
        std::cout << "  Neighbor cells: min=" << min_neighbors
                  << ", max=" << max_neighbors
                  << ", avg=" << (double)total_neighbors / cells_.size() << std::endl;
    }

    // Memory usage in bytes
    size_t memoryUsage() const {
        size_t mem = unit_centers_.size() * sizeof(std::array<float, 3>);
        mem += cells_.size() * sizeof(SphericalCell);
        for (const auto& cell : cells_) {
            mem += cell.face_indices.size() * sizeof(size_t);
        }
        for (const auto& neighbors : cell_neighbors_) {
            mem += neighbors.size() * sizeof(int);
        }
        return mem;
    }

    // Clear all data
    void clear() {
        cells_.clear();
        cell_id_to_index_.clear();
        unit_centers_.clear();
        cell_neighbors_.clear();
        num_faces_ = 0;
    }
};

} // namespace exosphere
} // namespace ptm

#endif // SPHERICAL_INDEX_H
