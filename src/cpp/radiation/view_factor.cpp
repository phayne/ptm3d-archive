#include "view_factor.h"
#include "../thermal/thermal_model.h"  // For ProgressBar

#include <embree4/rtcore.h>
#include <embree4/rtcore_ray.h>

#include <fstream>
#include <iostream>
#include <cstring>
#include <limits>
#include <atomic>

namespace radiation {

// ============================================================================
// ViewFactorMatrix Implementation
// ============================================================================

void ViewFactorMatrix::initialize(size_t n_faces, size_t estimated_nnz_per_row) {
    n_rows_ = n_faces;
    n_cols_ = n_faces;
    row_ptrs_.resize(n_faces + 1, 0);
    values_.reserve(n_faces * estimated_nnz_per_row);
    col_indices_.reserve(n_faces * estimated_nnz_per_row);
}

void ViewFactorMatrix::buildFromTriplets(size_t n_faces,
                                          const std::vector<int>& row_indices,
                                          const std::vector<int>& col_indices,
                                          const std::vector<double>& values) {
    n_rows_ = n_faces;
    n_cols_ = n_faces;
    size_t nnz = values.size();

    if (nnz == 0) {
        row_ptrs_.assign(n_faces + 1, 0);
        values_.clear();
        col_indices_.clear();
        return;
    }

    // Count entries per row
    std::vector<int> row_counts(n_faces, 0);
    for (size_t k = 0; k < nnz; ++k) {
        row_counts[row_indices[k]]++;
    }

    // Compute row pointers
    row_ptrs_.resize(n_faces + 1);
    row_ptrs_[0] = 0;
    for (size_t i = 0; i < n_faces; ++i) {
        row_ptrs_[i + 1] = row_ptrs_[i] + row_counts[i];
    }

    // Allocate values and column indices
    values_.resize(nnz);
    col_indices_.resize(nnz);

    // Fill in values (using row_counts as temporary write positions)
    std::fill(row_counts.begin(), row_counts.end(), 0);
    for (size_t k = 0; k < nnz; ++k) {
        int row = row_indices[k];
        int pos = row_ptrs_[row] + row_counts[row];
        values_[pos] = values[k];
        col_indices_[pos] = col_indices[k];
        row_counts[row]++;
    }

    // Sort each row by column index for efficient access
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < n_faces; ++i) {
        int start = row_ptrs_[i];
        int end = row_ptrs_[i + 1];
        if (end - start > 1) {
            // Simple insertion sort for small arrays
            for (int j = start + 1; j < end; ++j) {
                int col = col_indices_[j];
                double val = values_[j];
                int k = j - 1;
                while (k >= start && col_indices_[k] > col) {
                    col_indices_[k + 1] = col_indices_[k];
                    values_[k + 1] = values_[k];
                    k--;
                }
                col_indices_[k + 1] = col;
                values_[k + 1] = val;
            }
        }
    }
}

double ViewFactorMatrix::get(int row, int col) const {
    if (row < 0 || row >= static_cast<int>(n_rows_)) return 0.0;

    int start = row_ptrs_[row];
    int end = row_ptrs_[row + 1];

    // Binary search for column
    auto it = std::lower_bound(col_indices_.begin() + start,
                               col_indices_.begin() + end, col);
    if (it != col_indices_.begin() + end && *it == col) {
        return values_[it - col_indices_.begin()];
    }
    return 0.0;
}

void ViewFactorMatrix::getRow(int row, std::vector<int>& cols, std::vector<double>& vals) const {
    cols.clear();
    vals.clear();

    if (row < 0 || row >= static_cast<int>(n_rows_)) return;

    int start = row_ptrs_[row];
    int end = row_ptrs_[row + 1];

    cols.reserve(end - start);
    vals.reserve(end - start);

    for (int i = start; i < end; ++i) {
        cols.push_back(col_indices_[i]);
        vals.push_back(values_[i]);
    }
}

int ViewFactorMatrix::getRowNNZ(int row) const {
    if (row < 0 || row >= static_cast<int>(n_rows_)) return 0;
    return row_ptrs_[row + 1] - row_ptrs_[row];
}

void ViewFactorMatrix::matvec(const std::vector<double>& x, std::vector<double>& y) const {
    y.assign(n_rows_, 0.0);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n_rows_; ++i) {
        double sum = 0.0;
        int start = row_ptrs_[i];
        int end = row_ptrs_[i + 1];
        for (int k = start; k < end; ++k) {
            sum += values_[k] * x[col_indices_[k]];
        }
        y[i] = sum;
    }
}

void ViewFactorMatrix::matvecTranspose(const std::vector<double>& x, std::vector<double>& y) const {
    y.assign(n_cols_, 0.0);

    // This is less efficient but correct for sparse transpose
    for (size_t i = 0; i < n_rows_; ++i) {
        int start = row_ptrs_[i];
        int end = row_ptrs_[i + 1];
        for (int k = start; k < end; ++k) {
            #pragma omp atomic
            y[col_indices_[k]] += values_[k] * x[i];
        }
    }
}

std::vector<double> ViewFactorMatrix::rowSums() const {
    std::vector<double> sums(n_rows_, 0.0);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n_rows_; ++i) {
        double sum = 0.0;
        int start = row_ptrs_[i];
        int end = row_ptrs_[i + 1];
        for (int k = start; k < end; ++k) {
            sum += values_[k];
        }
        sums[i] = sum;
    }
    return sums;
}

void ViewFactorMatrix::clear() {
    values_.clear();
    col_indices_.clear();
    row_ptrs_.clear();
    n_rows_ = 0;
    n_cols_ = 0;
}

size_t ViewFactorMatrix::memoryUsage() const {
    return values_.size() * sizeof(double) +
           col_indices_.size() * sizeof(int) +
           row_ptrs_.size() * sizeof(int);
}

// Magic number for sparse view factor file format: "VFFM" = View Factor File Matrix
static constexpr uint32_t SPARSE_VF_MAGIC = 0x5646464D;
static constexpr uint32_t SPARSE_VF_VERSION = 1;

bool ViewFactorMatrix::writeBinary(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file) return false;

    // Write magic number and version for format detection
    file.write(reinterpret_cast<const char*>(&SPARSE_VF_MAGIC), sizeof(SPARSE_VF_MAGIC));
    file.write(reinterpret_cast<const char*>(&SPARSE_VF_VERSION), sizeof(SPARSE_VF_VERSION));

    // Write header
    uint64_t nrows = n_rows_;
    uint64_t ncols = n_cols_;
    uint64_t nnz = values_.size();

    file.write(reinterpret_cast<const char*>(&nrows), sizeof(uint64_t));
    file.write(reinterpret_cast<const char*>(&ncols), sizeof(uint64_t));
    file.write(reinterpret_cast<const char*>(&nnz), sizeof(uint64_t));

    // Write row pointers
    file.write(reinterpret_cast<const char*>(row_ptrs_.data()),
               row_ptrs_.size() * sizeof(int));

    // Write column indices
    file.write(reinterpret_cast<const char*>(col_indices_.data()),
               col_indices_.size() * sizeof(int));

    // Write values
    file.write(reinterpret_cast<const char*>(values_.data()),
               values_.size() * sizeof(double));

    return file.good();
}

bool ViewFactorMatrix::readBinary(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) return false;

    // Read and verify magic number and version
    uint32_t magic, version;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&version), sizeof(version));

    if (magic != SPARSE_VF_MAGIC) {
        std::cerr << "Invalid sparse view factor file format (magic: 0x"
                  << std::hex << magic << std::dec << ")" << std::endl;
        return false;
    }
    if (version != SPARSE_VF_VERSION) {
        std::cerr << "Unsupported sparse view factor file version: " << version << std::endl;
        return false;
    }

    // Read header
    uint64_t nrows, ncols, nnz;
    file.read(reinterpret_cast<char*>(&nrows), sizeof(uint64_t));
    file.read(reinterpret_cast<char*>(&ncols), sizeof(uint64_t));
    file.read(reinterpret_cast<char*>(&nnz), sizeof(uint64_t));

    n_rows_ = nrows;
    n_cols_ = ncols;

    // Read row pointers
    row_ptrs_.resize(nrows + 1);
    file.read(reinterpret_cast<char*>(row_ptrs_.data()),
              row_ptrs_.size() * sizeof(int));

    // Read column indices
    col_indices_.resize(nnz);
    file.read(reinterpret_cast<char*>(col_indices_.data()),
              col_indices_.size() * sizeof(int));

    // Read values
    values_.resize(nnz);
    file.read(reinterpret_cast<char*>(values_.data()),
              values_.size() * sizeof(double));

    return file.good();
}

// ============================================================================
// ViewFactorCalculator Implementation
// ============================================================================

ViewFactorCalculator::ViewFactorCalculator(const std::shared_ptr<ptm::Mesh>& mesh,
                                           RTCDevice device,
                                           RTCScene scene,
                                           const RadiationConfig& config)
    : mesh_(mesh), device_(device), scene_(scene), config_(config) {
    if (config_.verbose) {
        std::cout << "ViewFactorCalculator initialized with visibility_threshold="
                  << config_.visibility_threshold << std::endl;
    }
    precomputeFaceData();
}

ViewFactorCalculator::~ViewFactorCalculator() {
    // Note: We don't own the Embree device/scene, so don't release them
}

void ViewFactorCalculator::precomputeFaceData() {
    if (face_data_computed_) return;

    size_t n_faces = mesh_->faces.size();
    face_centers_.resize(n_faces);
    face_normals_.resize(n_faces);
    face_areas_.resize(n_faces);

    // Check if mesh already has pre-computed face data with outward-oriented normals
    // PTM3D's MeshUtils ensures normals point outward (away from body center)
    bool use_mesh_data = (mesh_->face_centers.size() == n_faces &&
                          mesh_->face_normals.size() == n_faces);

    if (use_mesh_data) {
        // Use pre-computed face centers and outward-oriented normals from mesh
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n_faces; ++i) {
            face_centers_[i] = mesh_->face_centers[i];
            face_normals_[i] = mesh_->face_normals[i];

            // Still need to compute areas
            const auto& face = mesh_->faces[i];
            const auto& v0 = mesh_->vertices[face[0]];
            const auto& v1 = mesh_->vertices[face[1]];
            const auto& v2 = mesh_->vertices[face[2]];
            auto e1 = subtract(v1, v0);
            auto e2 = subtract(v2, v0);
            auto n = cross(e1, e2);
            face_areas_[i] = length(n) / 2.0;
        }

        if (config_.verbose) {
            std::cout << "Using pre-computed face data from mesh (outward-oriented normals)" << std::endl;
        }
    } else {
        // Compute face data from scratch and orient normals outward
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n_faces; ++i) {
            const auto& face = mesh_->faces[i];
            const auto& v0 = mesh_->vertices[face[0]];
            const auto& v1 = mesh_->vertices[face[1]];
            const auto& v2 = mesh_->vertices[face[2]];

            // Compute center
            face_centers_[i] = {{
                (v0[0] + v1[0] + v2[0]) / 3.0,
                (v0[1] + v1[1] + v2[1]) / 3.0,
                (v0[2] + v1[2] + v2[2]) / 3.0
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

            // Ensure outward-pointing normals (away from body center at origin)
            // This matches PTM3D convention: normal · center > 0 means outward
            const auto& c = face_centers_[i];
            double radial_dot = face_normals_[i][0] * c[0] +
                                face_normals_[i][1] * c[1] +
                                face_normals_[i][2] * c[2];
            if (radial_dot < 0) {
                // Flip to point outward
                face_normals_[i][0] = -face_normals_[i][0];
                face_normals_[i][1] = -face_normals_[i][1];
                face_normals_[i][2] = -face_normals_[i][2];
            }
        }

        if (config_.verbose) {
            std::cout << "Computed face data with outward-oriented normals" << std::endl;
        }
    }

    face_data_computed_ = true;

    // Build spatial grid
    double avg_area = 0.0;
    for (double a : face_areas_) avg_area += a;
    avg_area /= n_faces;
    double cell_hint = std::sqrt(avg_area) * 10.0;
    spatial_grid_.build(face_centers_, cell_hint);
    spatial_grid_built_ = true;
}

double ViewFactorCalculator::computeFaceArea(int face_idx) {
    if (face_data_computed_) {
        return face_areas_[face_idx];
    }
    const auto& face = mesh_->faces[face_idx];
    const auto& v0 = mesh_->vertices[face[0]];
    const auto& v1 = mesh_->vertices[face[1]];
    const auto& v2 = mesh_->vertices[face[2]];
    auto e1 = subtract(v1, v0);
    auto e2 = subtract(v2, v0);
    return length(cross(e1, e2)) / 2.0;
}

ViewFactorMatrix ViewFactorCalculator::computeAll() {
    std::vector<int> all_faces(mesh_->faces.size());
    std::iota(all_faces.begin(), all_faces.end(), 0);
    return computeSubset(all_faces);
}

// Static debug counters for view factor computation (centroid method)
static std::atomic<int> g_angle_reject{0};
static std::atomic<int> g_vis_reject{0};
static std::atomic<int> g_accepted{0};
static std::atomic<bool> g_first_accept_printed{false};

// Static debug counters for subdivision method
static std::atomic<int> g_subdiv_angle_reject{0};
static std::atomic<int> g_subdiv_vis_reject{0};
static std::atomic<int> g_subdiv_accepted{0};
static std::atomic<int> g_subdiv_weight_zero{0};

ViewFactorMatrix ViewFactorCalculator::computeSubset(const std::vector<int>& face_indices) {
    size_t n_subset = face_indices.size();
    size_t n_total = mesh_->faces.size();

    if (config_.verbose) {
        std::cout << "Computing view factors for " << n_subset << " faces..." << std::endl;
    }

    // Compute search radius based on mesh extent, not just face size.
    // For crater/terrain meshes, we need to search far enough to find
    // faces on opposite walls that can exchange radiation.
    //
    // Use the larger of:
    // 1. search_radius_factor * sqrt(avg_area) - for fine meshes
    // 2. mesh_extent / 2 - to ensure we can reach across the crater
    double avg_area = 0.0;
    for (int idx : face_indices) {
        avg_area += face_areas_[idx];
    }
    avg_area /= n_subset;
    double face_based_radius = config_.search_radius_factor * std::sqrt(avg_area);

    // Compute mesh extent from bounding box
    double min_x = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::lowest();
    double min_y = std::numeric_limits<double>::max();
    double max_y = std::numeric_limits<double>::lowest();
    double min_z = std::numeric_limits<double>::max();
    double max_z = std::numeric_limits<double>::lowest();

    for (int idx : face_indices) {
        const auto& c = face_centers_[idx];
        min_x = std::min(min_x, c[0]); max_x = std::max(max_x, c[0]);
        min_y = std::min(min_y, c[1]); max_y = std::max(max_y, c[1]);
        min_z = std::min(min_z, c[2]); max_z = std::max(max_z, c[2]);
    }

    double extent_x = max_x - min_x;
    double extent_y = max_y - min_y;
    double extent_z = max_z - min_z;
    double mesh_extent = std::sqrt(extent_x * extent_x + extent_y * extent_y + extent_z * extent_z);
    double extent_based_radius = mesh_extent;  // Full diagonal to reach any face

    double search_radius = std::max(face_based_radius, extent_based_radius);

    if (config_.verbose) {
        std::cout << "Search radius: " << search_radius << " m "
                  << "(face-based: " << face_based_radius << ", extent-based: " << extent_based_radius << ")"
                  << std::endl;
    }

    // Collect triplets in thread-local storage
    std::vector<std::vector<int>> local_rows(omp_get_max_threads());
    std::vector<std::vector<int>> local_cols(omp_get_max_threads());
    std::vector<std::vector<double>> local_vals(omp_get_max_threads());

    // Progress tracking with atomic counter for thread-safe updates
    // Always show progress bar (not gated by verbose) since this is a long operation
    std::atomic<size_t> progress_counter{0};
    const size_t progress_interval = std::max(n_subset / 100, static_cast<size_t>(1));  // Update ~100 times
    thermal::ProgressBar progress_bar(static_cast<int>(n_subset), "View factor computation: ", 50, true);
    // Show initial 0% progress immediately
    progress_bar.update(0);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        local_rows[tid].reserve(n_subset * 50 / omp_get_max_threads());
        local_cols[tid].reserve(n_subset * 50 / omp_get_max_threads());
        local_vals[tid].reserve(n_subset * 50 / omp_get_max_threads());

        #pragma omp for schedule(dynamic, 16)
        for (size_t i = 0; i < n_subset; ++i) {
            int face_i = face_indices[i];

            // Find neighbor faces
            auto neighbors = findNeighbors(face_i, search_radius);

            // Limit number of neighbors if needed
            // Sort by POTENTIAL VIEW FACTOR (A_j * cos_i * cos_j / r²), not distance.
            // This ensures we keep the faces that contribute most to radiation exchange.
            if (static_cast<int>(neighbors.size()) > config_.max_neighbors) {
                std::vector<std::pair<double, int>> potential_vf;
                potential_vf.reserve(neighbors.size());

                const auto& c_i = face_centers_[face_i];
                const auto& n_i = face_normals_[face_i];

                for (int j : neighbors) {
                    const auto& c_j = face_centers_[j];
                    const auto& n_j = face_normals_[j];
                    double A_j = face_areas_[j];

                    auto r_vec = subtract(c_j, c_i);
                    double r2 = dot(r_vec, r_vec);
                    if (r2 < 1e-20) {
                        potential_vf.push_back({0.0, j});
                        continue;
                    }

                    double r = std::sqrt(r2);
                    std::array<double, 3> r_hat = {{r_vec[0]/r, r_vec[1]/r, r_vec[2]/r}};

                    // Use absolute cosines (works for any normal convention)
                    double cos_i = std::abs(dot(n_i, r_hat));
                    double cos_j = std::abs(dot(n_j, r_hat));

                    // Potential view factor (higher is better, so we negate for sorting)
                    double potential = A_j * cos_i * cos_j / r2;
                    potential_vf.push_back({-potential, j});  // Negative so we keep highest
                }

                // Sort by potential (highest first due to negation)
                std::partial_sort(potential_vf.begin(),
                                  potential_vf.begin() + config_.max_neighbors,
                                  potential_vf.end());
                neighbors.resize(config_.max_neighbors);
                for (int k = 0; k < config_.max_neighbors; ++k) {
                    neighbors[k] = potential_vf[k].second;
                }
            }

            // Compute view factors to neighbors
            // Debug counters (per thread)
            int dbg_threshold_reject = 0;
            int dbg_accepted = 0;

            double max_F = 0.0;
            for (int face_j : neighbors) {
                if (face_j == face_i) continue;  // Skip self

                double F_ij = computeAdaptive(face_i, face_j);
                max_F = std::max(max_F, F_ij);

                if (F_ij > config_.visibility_threshold) {
                    local_rows[tid].push_back(face_i);
                    local_cols[tid].push_back(face_j);
                    local_vals[tid].push_back(F_ij);
                    dbg_accepted++;
                } else if (F_ij > 0) {
                    dbg_threshold_reject++;
                }
            }

            // Update progress counter (all threads increment)
            size_t current = ++progress_counter;

            // Only one thread updates the display to avoid contention
            // Use modulo on current count so updates happen regularly regardless of which thread
            if (current % progress_interval == 0) {
                #pragma omp critical(progress_update)
                {
                    progress_bar.update(static_cast<int>(current));
                }
            }
        }
    }

    // Finalize progress bar
    progress_bar.finish();

    // Print debug summary after progress bar completes (only in verbose mode)
    if (config_.verbose) {
        std::cout << "View factor computation complete." << std::endl;
    }

    // Merge thread-local data
    std::vector<int> all_rows, all_cols;
    std::vector<double> all_vals;

    size_t total_nnz = 0;
    for (int t = 0; t < omp_get_max_threads(); ++t) {
        total_nnz += local_rows[t].size();
    }
    all_rows.reserve(total_nnz);
    all_cols.reserve(total_nnz);
    all_vals.reserve(total_nnz);

    for (int t = 0; t < omp_get_max_threads(); ++t) {
        all_rows.insert(all_rows.end(), local_rows[t].begin(), local_rows[t].end());
        all_cols.insert(all_cols.end(), local_cols[t].begin(), local_cols[t].end());
        all_vals.insert(all_vals.end(), local_vals[t].begin(), local_vals[t].end());
    }

    // Build sparse matrix
    ViewFactorMatrix F;
    F.buildFromTriplets(n_total, all_rows, all_cols, all_vals);

    if (config_.verbose) {
        std::cout << "View factor matrix: " << F.numRows() << " x " << F.numCols()
                  << ", " << F.numNonZeros() << " non-zeros ("
                  << (100.0 * F.numNonZeros()) / (F.numRows() * F.numCols())
                  << "% dense)" << std::endl;
        std::cout << "Memory usage: " << F.memoryUsage() / (1024.0 * 1024.0)
                  << " MB" << std::endl;

        // Print debug rejection statistics
        std::cout << "View factor rejection statistics (centroid method):" << std::endl;
        std::cout << "  Rejected by angle check: " << g_angle_reject.load() << std::endl;
        std::cout << "  Rejected by visibility: " << g_vis_reject.load() << std::endl;
        std::cout << "  Accepted: " << g_accepted.load() << std::endl;

        std::cout << "View factor rejection statistics (subdivision method):" << std::endl;
        std::cout << "  Rejected (weight_sum=0, all angles bad): " << g_subdiv_weight_zero.load() << std::endl;
        std::cout << "  Rejected by visibility: " << g_subdiv_vis_reject.load() << std::endl;
        std::cout << "  Accepted: " << g_subdiv_accepted.load() << std::endl;

        // Reset for next run
        g_angle_reject = 0;
        g_vis_reject = 0;
        g_accepted = 0;
        g_first_accept_printed = false;
        g_subdiv_weight_zero = 0;
        g_subdiv_vis_reject = 0;
        g_subdiv_accepted = 0;

        // Diagnostic: Check row sum variation (indicates view factor consistency)
        auto row_sums = F.rowSums();
        double min_sum = 1e9, max_sum = 0.0, sum_total = 0.0;
        int zero_rows = 0;
        for (size_t i = 0; i < row_sums.size(); ++i) {
            if (row_sums[i] > 0) {
                min_sum = std::min(min_sum, row_sums[i]);
                max_sum = std::max(max_sum, row_sums[i]);
                sum_total += row_sums[i];
            } else {
                zero_rows++;
            }
        }
        size_t nonzero_rows = row_sums.size() - zero_rows;
        double mean_sum = (nonzero_rows > 0) ? sum_total / nonzero_rows : 0.0;
        std::cout << "Row sum statistics (terrain view factors):" << std::endl;
        std::cout << "  Zero rows (no view factors): " << zero_rows << std::endl;
        std::cout << "  Non-zero rows: " << nonzero_rows << std::endl;
        std::cout << "  Row sum range: [" << min_sum << ", " << max_sum << "]" << std::endl;
        std::cout << "  Mean row sum: " << mean_sum << std::endl;

        // Check variation in row sums for neighboring faces
        // High variation among neighbors indicates potential speckling source
        double max_neighbor_variation = 0.0;
        int high_var_face = -1;
        for (size_t i = 0; i < n_total && i < 5000; ++i) {
            if (row_sums[i] == 0) continue;
            // Check adjacent faces (assuming roughly sequential numbering)
            for (int offset = 1; offset <= 5; ++offset) {
                if (i + offset < n_total && row_sums[i + offset] > 0) {
                    double ratio = row_sums[i] / row_sums[i + offset];
                    if (ratio < 1.0) ratio = 1.0 / ratio;
                    if (ratio > max_neighbor_variation) {
                        max_neighbor_variation = ratio;
                        high_var_face = static_cast<int>(i);
                    }
                }
            }
        }
        if (high_var_face >= 0) {
            std::cout << "  Max neighbor row sum ratio: " << max_neighbor_variation
                      << " at face " << high_var_face << std::endl;
            std::cout << "    Face " << high_var_face << " row sum: " << row_sums[high_var_face] << std::endl;
            for (int offset = 1; offset <= 3 && high_var_face + offset < static_cast<int>(n_total); ++offset) {
                std::cout << "    Face " << (high_var_face + offset) << " row sum: "
                          << row_sums[high_var_face + offset] << std::endl;
            }
        }
    }

    return F;
}

double ViewFactorCalculator::computePair(int face_i, int face_j) {
    return computeAdaptive(face_i, face_j);
}

double ViewFactorCalculator::computeSkyViewFactor(int face_idx) {
    // Sky view factor = 1 - sum of terrain view factors
    // For now, compute by summing all view factors to other faces
    double terrain_sum = 0.0;

    // Find all neighbors within search radius
    double avg_area = 0.0;
    for (double a : face_areas_) avg_area += a;
    avg_area /= face_areas_.size();
    double search_radius = config_.search_radius_factor * std::sqrt(avg_area);

    auto neighbors = findNeighbors(face_idx, search_radius);

    for (int j : neighbors) {
        if (j == face_idx) continue;
        terrain_sum += computeAdaptive(face_idx, j);
    }

    return std::max(0.0, 1.0 - terrain_sum);
}

double ViewFactorCalculator::computeCentroid(int face_i, int face_j) {
    // M1: Centroid approximation
    // F_ij ≈ (A_j / π) * (cos θ_i * cos θ_j) / r²
    //
    // For two faces to exchange radiation:
    // - Face i must "look toward" face j: the component of n_i along r_ij should
    //   have the right sign. For outward normals, this means dot(n_i, r_hat) > 0.
    //   For inward normals, it would be dot(n_i, r_hat) < 0.
    // - Similarly for face j looking back at face i.
    //
    // To handle arbitrary normal orientations, we check if the faces are
    // geometrically oriented toward each other: normals should point in
    // "opposite" directions relative to the connecting vector.
    // This is captured by: dot(n_i, r_hat) * dot(n_j, r_hat) < 0

    const auto& c_i = face_centers_[face_i];
    const auto& c_j = face_centers_[face_j];
    const auto& n_i = face_normals_[face_i];
    const auto& n_j = face_normals_[face_j];
    double A_j = face_areas_[face_j];

    // Vector from i to j
    auto r_vec = subtract(c_j, c_i);
    double r2 = dot(r_vec, r_vec);
    if (r2 < 1e-20) return 0.0;

    double r = std::sqrt(r2);
    std::array<double, 3> r_hat = {{r_vec[0] / r, r_vec[1] / r, r_vec[2] / r}};

    // Compute dot products with connecting vector r_hat (pointing from i to j)
    double dot_i = dot(n_i, r_hat);   // n_i · r_hat
    double dot_j = dot(n_j, r_hat);   // n_j · r_hat

    // View factor formula with OUTWARD normals (PTM3D convention):
    //
    // For face i to "see" face j:
    //   - Face i's outward normal should have a component toward face j
    //     → cos_i = n_i · r_hat > 0 (face i looks toward j)
    //   - Face j's outward normal should have a component toward face i
    //     → cos_j = n_j · (-r_hat) = -dot_j > 0 (face j looks toward i)
    //
    // For mutual visibility: dot_i > 0 AND dot_j < 0
    //
    // The view factor formula is:
    //   F_ij = A_j * cos_i * cos_j / (π * r²)
    // where cos_i = dot_i and cos_j = -dot_j (both positive for visible faces)

    double cos_i = dot_i;       // n_i · r_hat (positive if i faces toward j)
    double cos_j = -dot_j;      // n_j · (-r_hat) (positive if j faces toward i)

    // Skip nearly-grazing angles (avoid numerical issues)
    if (cos_i < 1e-6 || cos_j < 1e-6) {
        g_angle_reject++;
        return 0.0;
    }

    // Check visibility using ray tracing
    if (!isVisible(c_i, c_j)) {
        g_vis_reject++;
        return 0.0;
    }

    // View factor formula
    double F_ij = (A_j / M_PI) * (cos_i * cos_j) / r2;

    g_accepted++;

    return std::max(0.0, F_ij);
}

double ViewFactorCalculator::computeSubdivision(int face_i, int face_j, int level) {
    // M2: Single-point to area integration method
    // Use centroid of face_i, but integrate over subdivided face_j
    // This is more accurate than M1 for nearby faces and more stable than M3
    //
    // F_ij = (1/π) * ∫∫_{A_j} (cos_i * cos_j / r²) dA_j
    //
    // We approximate this by summing over subdivision points of face_j,
    // where each point represents an equal area element A_j/n_points.

    std::vector<std::array<double, 3>> points_j;
    std::vector<double> weights_j;

    getSubdivisionPoints(face_j, level, points_j, weights_j);

    const auto& c_i = face_centers_[face_i];
    const auto& n_i = face_normals_[face_i];
    const auto& n_j = face_normals_[face_j];
    double A_j = face_areas_[face_j];

    // First check center-to-center visibility
    if (!isVisible(c_i, face_centers_[face_j])) {
        g_subdiv_vis_reject++;
        return 0.0;
    }

    double F_sum = 0.0;
    int n_valid = 0;
    int angle_rejects = 0;

    // Integrate over receiving face (face_j)
    for (size_t pj = 0; pj < points_j.size(); ++pj) {
        const auto& p_j = points_j[pj];

        auto r_vec = subtract(p_j, c_i);
        double r2 = dot(r_vec, r_vec);
        if (r2 < 1e-20) continue;

        double r = std::sqrt(r2);
        std::array<double, 3> r_hat = {{r_vec[0] / r, r_vec[1] / r, r_vec[2] / r}};

        // Compute dot products with connecting vector r_hat (pointing from i to j)
        double dot_i = dot(n_i, r_hat);
        double dot_j = dot(n_j, r_hat);

        // With outward normals:
        //   cos_i = dot_i (positive if face i looks toward point)
        //   cos_j = -dot_j (positive if face j looks toward face i)
        double cos_i = dot_i;
        double cos_j = -dot_j;

        // Skip faces that don't face each other (cos < 0) or nearly-grazing angles
        if (cos_i < 1e-6 || cos_j < 1e-6) {
            angle_rejects++;
            continue;
        }

        // Each subdivision point represents equal area A_j / n_points
        // F contribution = (cos_i * cos_j / r²) * (A_j / n_points) / π
        F_sum += (cos_i * cos_j) / r2;
        n_valid++;
    }

    if (n_valid == 0) {
        g_subdiv_weight_zero++;
        return 0.0;
    }

    // Final formula: F_ij = (A_j / π) * (1/n_points) * sum
    // where sum = Σ (cos_i * cos_j / r²)
    double n_points = static_cast<double>(points_j.size());
    double F_ij = (A_j / M_PI) * F_sum / n_points;

    // Debug: print first few computed view factors with detailed cosine info
    if (F_ij > 0) {
        g_subdiv_accepted++;
    }
    return std::max(0.0, F_ij);
}

double ViewFactorCalculator::computeAdaptive(int face_i, int face_j) {
    // Adaptive method selection based on dimensionless separation Ns
    // Ns = r / sqrt(A) where r is distance and A is face area

    const auto& c_i = face_centers_[face_i];
    const auto& c_j = face_centers_[face_j];

    auto r_vec = subtract(c_j, c_i);
    double r = std::sqrt(dot(r_vec, r_vec));

    double A_eff = std::sqrt(face_areas_[face_i] * face_areas_[face_j]);
    double Ns = r / std::sqrt(A_eff);

    if (Ns > config_.adaptive_ns_threshold) {
        // Far apart: use centroid method
        return computeCentroid(face_i, face_j);
    } else {
        // Close: use subdivision method
        return computeSubdivision(face_i, face_j, config_.subdivision_level);
    }
}

bool ViewFactorCalculator::isVisible(const std::array<double, 3>& p1,
                                     const std::array<double, 3>& p2) {
    // Ray from p1 toward p2
    auto dir = subtract(p2, p1);
    double dist = length(dir);
    if (dist < 1e-10) return true;  // Same point

    // Normalize direction
    dir = {{dir[0] / dist, dir[1] / dist, dir[2] / dist}};

    // Use BOTH relative and absolute offsets to avoid self-intersection.
    // The relative offset handles long-distance rays, while the absolute
    // offset (based on typical face size) handles short-distance rays
    // where a tiny relative offset might still hit the source face.
    //
    // For crater meshes with ~20m face edges, use 0.5m as minimum offset.
    double relative_offset = dist * 0.001;  // 0.1% of distance
    double absolute_offset = 0.5;           // 0.5m minimum
    double offset = std::max(relative_offset, absolute_offset);

    // Clamp offset to at most 10% of distance to avoid overshooting for nearby faces
    offset = std::min(offset, dist * 0.1);

    std::array<double, 3> origin = {{
        p1[0] + offset * dir[0],
        p1[1] + offset * dir[1],
        p1[2] + offset * dir[2]
    }};

    // Similarly, stop the ray before reaching p2 with adequate margin
    double end_margin = std::max(dist * 0.001, absolute_offset);
    end_margin = std::min(end_margin, dist * 0.1);
    double tfar = dist - offset - end_margin;

    // If the effective ray length is too short, consider it visible
    // (faces are essentially touching)
    if (tfar < 0.1) return true;

    // Create ray
    RTCRayHit rayhit;
    rayhit.ray.org_x = static_cast<float>(origin[0]);
    rayhit.ray.org_y = static_cast<float>(origin[1]);
    rayhit.ray.org_z = static_cast<float>(origin[2]);
    rayhit.ray.dir_x = static_cast<float>(dir[0]);
    rayhit.ray.dir_y = static_cast<float>(dir[1]);
    rayhit.ray.dir_z = static_cast<float>(dir[2]);
    rayhit.ray.tnear = 0.0f;
    rayhit.ray.tfar = static_cast<float>(tfar);
    rayhit.ray.mask = 0xFFFFFFFF;
    rayhit.ray.flags = 0;
    rayhit.ray.time = 0.0f;
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rayhit.hit.primID = RTC_INVALID_GEOMETRY_ID;
    rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

    // Trace ray
    rtcIntersect1(scene_, &rayhit);

    // Visible if no hit (geomID remains invalid)
    return rayhit.hit.geomID == RTC_INVALID_GEOMETRY_ID;
}

std::vector<bool> ViewFactorCalculator::checkVisibilityBatch(
    const std::vector<std::array<double, 3>>& origins,
    const std::vector<std::array<double, 3>>& targets) {

    size_t n = origins.size();
    std::vector<bool> visible(n, false);

    // Process in batches for cache efficiency
    constexpr size_t BATCH_SIZE = 64;

    for (size_t start = 0; start < n; start += BATCH_SIZE) {
        size_t end = std::min(start + BATCH_SIZE, n);

        for (size_t i = start; i < end; ++i) {
            visible[i] = isVisible(origins[i], targets[i]);
        }
    }

    return visible;
}

std::vector<int> ViewFactorCalculator::findNeighbors(int face_idx, double radius) {
    if (!spatial_grid_built_) {
        // Fallback: linear search (slow)
        std::vector<int> neighbors;
        const auto& c = face_centers_[face_idx];
        double r2 = radius * radius;

        for (size_t j = 0; j < face_centers_.size(); ++j) {
            auto d = subtract(face_centers_[j], c);
            if (dot(d, d) <= r2) {
                neighbors.push_back(static_cast<int>(j));
            }
        }
        return neighbors;
    }

    return spatial_grid_.query(face_centers_[face_idx], radius);
}

void ViewFactorCalculator::getSubdivisionPoints(int face_idx, int level,
                                                 std::vector<std::array<double, 3>>& points,
                                                 std::vector<double>& weights) {
    const auto& face = mesh_->faces[face_idx];
    const auto& v0 = mesh_->vertices[face[0]];
    const auto& v1 = mesh_->vertices[face[1]];
    const auto& v2 = mesh_->vertices[face[2]];

    // Generate barycentric coordinates for subdivision
    // Level 0: 1 point (centroid)
    // Level 1: 4 points
    // Level 2: 16 points, etc.

    int n_per_edge = 1 << level;  // 2^level
    int n_points = 0;

    // Count total points
    for (int i = 0; i <= n_per_edge; ++i) {
        for (int j = 0; j <= n_per_edge - i; ++j) {
            n_points++;
        }
    }

    points.resize(n_points);
    weights.resize(n_points, 1.0 / n_points);

    int idx = 0;
    double step = 1.0 / n_per_edge;

    for (int i = 0; i <= n_per_edge; ++i) {
        double u = i * step;
        for (int j = 0; j <= n_per_edge - i; ++j) {
            double v = j * step;
            double w = 1.0 - u - v;

            points[idx] = {{
                u * v0[0] + v * v1[0] + w * v2[0],
                u * v0[1] + v * v1[1] + w * v2[1],
                u * v0[2] + v * v1[2] + w * v2[2]
            }};
            idx++;
        }
    }
}

// Vector utilities
std::array<double, 3> ViewFactorCalculator::subtract(const std::array<double, 3>& a,
                                                      const std::array<double, 3>& b) {
    return {{a[0] - b[0], a[1] - b[1], a[2] - b[2]}};
}

std::array<double, 3> ViewFactorCalculator::cross(const std::array<double, 3>& a,
                                                   const std::array<double, 3>& b) {
    return {{
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    }};
}

double ViewFactorCalculator::dot(const std::array<double, 3>& a,
                                  const std::array<double, 3>& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

double ViewFactorCalculator::length(const std::array<double, 3>& v) {
    return std::sqrt(dot(v, v));
}

std::array<double, 3> ViewFactorCalculator::normalize(const std::array<double, 3>& v) {
    double len = length(v);
    if (len < 1e-20) return {{0.0, 0.0, 1.0}};
    return {{v[0] / len, v[1] / len, v[2] / len}};
}

// ============================================================================
// SpatialGrid Implementation
// ============================================================================

void ViewFactorCalculator::SpatialGrid::build(
    const std::vector<std::array<double, 3>>& centers, double cell_size_hint) {

    if (centers.empty()) return;

    // Find bounding box
    min_bound = centers[0];
    max_bound = centers[0];

    for (const auto& c : centers) {
        for (int d = 0; d < 3; ++d) {
            min_bound[d] = std::min(min_bound[d], c[d]);
            max_bound[d] = std::max(max_bound[d], c[d]);
        }
    }

    // Add small margin
    for (int d = 0; d < 3; ++d) {
        double margin = (max_bound[d] - min_bound[d]) * 0.01 + cell_size_hint;
        min_bound[d] -= margin;
        max_bound[d] += margin;
    }

    // Determine cell size and grid dimensions
    for (int d = 0; d < 3; ++d) {
        double extent = max_bound[d] - min_bound[d];
        grid_dims[d] = std::max(1, static_cast<int>(extent / cell_size_hint) + 1);
        cell_size[d] = extent / grid_dims[d];
    }

    // Allocate cells
    int total_cells = grid_dims[0] * grid_dims[1] * grid_dims[2];
    cells.resize(total_cells);

    // Insert centers into cells
    for (size_t i = 0; i < centers.size(); ++i) {
        int cell_idx = getCellIndex(centers[i]);
        if (cell_idx >= 0 && cell_idx < total_cells) {
            cells[cell_idx].push_back(static_cast<int>(i));
        }
    }
}

std::vector<int> ViewFactorCalculator::SpatialGrid::query(
    const std::array<double, 3>& center, double radius) const {

    std::vector<int> result;
    double r2 = radius * radius;

    // Find cell range to search
    std::array<int, 3> min_cell, max_cell;
    for (int d = 0; d < 3; ++d) {
        min_cell[d] = std::max(0, static_cast<int>((center[d] - radius - min_bound[d]) / cell_size[d]));
        max_cell[d] = std::min(grid_dims[d] - 1,
                               static_cast<int>((center[d] + radius - min_bound[d]) / cell_size[d]));
    }

    // Search relevant cells
    for (int iz = min_cell[2]; iz <= max_cell[2]; ++iz) {
        for (int iy = min_cell[1]; iy <= max_cell[1]; ++iy) {
            for (int ix = min_cell[0]; ix <= max_cell[0]; ++ix) {
                int cell_idx = ix + iy * grid_dims[0] + iz * grid_dims[0] * grid_dims[1];
                // Note: We return all faces in range; caller will check actual distance
                result.insert(result.end(), cells[cell_idx].begin(), cells[cell_idx].end());
            }
        }
    }

    return result;
}

int ViewFactorCalculator::SpatialGrid::getCellIndex(const std::array<double, 3>& point) const {
    int ix = static_cast<int>((point[0] - min_bound[0]) / cell_size[0]);
    int iy = static_cast<int>((point[1] - min_bound[1]) / cell_size[1]);
    int iz = static_cast<int>((point[2] - min_bound[2]) / cell_size[2]);

    ix = std::max(0, std::min(ix, grid_dims[0] - 1));
    iy = std::max(0, std::min(iy, grid_dims[1] - 1));
    iz = std::max(0, std::min(iz, grid_dims[2] - 1));

    return ix + iy * grid_dims[0] + iz * grid_dims[0] * grid_dims[1];
}

} // namespace radiation
