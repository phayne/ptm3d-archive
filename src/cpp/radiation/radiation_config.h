#pragma once

#include <vector>
#include <cmath>

namespace radiation {

// Configuration for radiation exchange computation
struct RadiationConfig {
    // Master enable/disable
    bool enabled = false;

    // Face selection criteria
    double min_shadow_fraction = 0.3;    // Min fraction of time in shadow to consider

    // View factor computation parameters
    double visibility_threshold = 1e-8;  // Min F_ij value to store (smaller = zero)
    int max_neighbors = 500;             // Hard ceiling: max neighbors per face (reduced from 2000 for better performance)
    double search_radius_factor = 10.0;  // Search radius in units of sqrt(face_area)

    // Adaptive neighbor selection (per-face variable neighbor count)
    bool adaptive_neighbors = true;      // Enable per-face adaptive selection based on captured energy
    double captured_energy_target = 0.95; // Target fraction of radiative potential to capture (95%)
    int min_neighbors = 20;              // Minimum neighbors per face (ensures basic coverage)

    // Cluster-level visibility pre-cull (conservative octree cull before per-face ranking).
    // Prunes candidate neighbors whose cluster's average normal is strongly back-facing.
    // Threshold matches the H-matrix cluster cull (cos > 0.5).
    bool use_cluster_cull = true;

    // View factor method selection
    enum class ViewFactorMethod {
        CENTROID,       // M1: Fast centroid approximation
        SUBDIVISION,    // M3: Subdivision for accuracy
        ADAPTIVE        // Auto-select based on face separation
    };
    ViewFactorMethod method = ViewFactorMethod::ADAPTIVE;

    // Subdivision parameters (for M2/M3 method)
    int subdivision_level = 3;           // Subdivision levels for receiving face (10 points at level 3)
    double adaptive_ns_threshold = 5.0;  // Ns threshold: use M2 when r/sqrt(A) < 5

    // Radiosity solver parameters
    int max_iterations = 100;            // Max Jacobi iterations
    double convergence_tolerance = 0.01; // Relative convergence criterion
    bool use_acceleration = true;        // Use Gauss-Seidel acceleration

    // Multi-scattering control
    int max_bounces = 1;                 // Number of scattering bounces (1 = single)

    // Performance options
    bool parallel_view_factors = true;   // Parallelize view factor computation
    bool cache_view_factors = true;      // Cache view factors between timesteps
    int visibility_batch_size = 1024;    // Batch size for visibility tests

    // GPU acceleration options
    bool use_gpu = false;                // Use GPU for radiosity solver (requires OpenCL)

    // H-matrix compression options (for large meshes)
    bool use_hmatrix = false;            // Use hierarchical matrix compression
    // Bypass the n_faces > 1M auto-disable so H-matrix + OOC is attempted regardless.
    // Intended for the probe run that determines whether the >1M disable is still
    // warranted; left off by default so existing behaviour is unchanged.
    bool force_hmatrix = false;
    double hmatrix_tolerance = 1e-2;     // SVD truncation tolerance for low-rank blocks
    int hmatrix_max_depth = 10;          // Maximum quadtree depth
    int hmatrix_min_block_size = 64;     // Minimum block size before treating as leaf
    int hmatrix_max_rank = 50;           // Maximum rank for low-rank blocks

    // ACA (Adaptive Cross Approximation) options - much faster than SVD for large meshes
    bool use_aca = true;                 // Use ACA by default (50-100x faster than SVD)
    double aca_tolerance = 1e-3;         // ACA relative tolerance for convergence
    int aca_max_rank = 50;               // Maximum rank before falling back to sparse

    // Thread control for H-matrix building
    int hmatrix_threads = 0;             // 0 = auto, >0 = manual thread limit

    // Output control
    bool output_view_factors = false;    // Write view factor matrix to file
    bool output_indirect_flux = false;   // Write indirect flux per face
    bool verbose = false;                // Print progress information
};

// Face classification for selective radiation exchange
struct FaceClassification {
    // Primary classification
    std::vector<bool> needs_exchange;       // True if face needs radiation exchange
    std::vector<int> exchange_face_indices; // Global indices of exchange faces

    // Diagnostic metrics per face
    std::vector<double> shadow_fractions;   // Fraction of time in shadow [0,1]
    std::vector<double> min_temperatures;   // Minimum surface temperature [K]

    // Summary statistics
    size_t total_faces = 0;
    size_t exchange_faces = 0;
    double mean_shadow_fraction = 0.0;

    // Clear all data
    void clear() {
        needs_exchange.clear();
        exchange_face_indices.clear();
        shadow_fractions.clear();
        min_temperatures.clear();
        total_faces = 0;
        exchange_faces = 0;
    }

    // Reserve memory for N faces
    void reserve(size_t n) {
        needs_exchange.reserve(n);
        shadow_fractions.reserve(n);
        min_temperatures.reserve(n);
    }

    // Resize to N faces
    void resize(size_t n) {
        needs_exchange.resize(n, false);
        shadow_fractions.resize(n, 0.0);
        min_temperatures.resize(n, 0.0);
        total_faces = n;
    }

    // Update exchange face list and statistics
    void finalize() {
        exchange_face_indices.clear();
        double sum_shadow = 0.0;

        for (size_t i = 0; i < needs_exchange.size(); ++i) {
            if (needs_exchange[i]) {
                exchange_face_indices.push_back(static_cast<int>(i));
            }
            sum_shadow += shadow_fractions[i];
        }

        exchange_faces = exchange_face_indices.size();
        if (total_faces > 0) {
            mean_shadow_fraction = sum_shadow / total_faces;
        }
    }
};

} // namespace radiation
