#pragma once

#include <vector>
#include <cmath>

namespace radiation {

// Configuration for radiation exchange computation
struct RadiationConfig {
    // Master enable/disable
    bool enabled = false;

    // ---- Radiation-exchange face set (PTM-98) --------------------------------
    // Which faces keep the indirect flux the radiosity solve already computed
    // for them. RadiositySolver::solve() returns Q_indirect for ALL n faces
    // (radiation_exchange.cpp, "Compute absorbed indirect flux at each face");
    // this set is applied AFTER the solve as a copy mask, so it never changes
    // the solve, only which computed values survive.
    //
    //   All (default) — every face keeps its flux. Window-INDEPENDENT by
    //       construction, and the only setting with a physical justification:
    //       the radiosity solution is a global self-consistent linear solve, so
    //       zeroing part of its output is data loss, not an approximation.
    //   Shadow — legacy gate: keep only faces in shadow at least
    //       min_shadow_fraction of the time. WINDOW-DEPENDENT and therefore
    //       chunk-decomposition dependent (PTM-98): the shadow fraction is
    //       measured over the run's own illumination series, whose length is
    //       n_equilibration_steps + n_output_steps. A warm-started chunk has no
    //       equilibration samples, so a short chunk that happens to fall in
    //       local daytime silently drops faces out of radiation exchange
    //       entirely — measured 82.38 K / 6367 faces > 1 K on a 3-chunk
    //       lunation, 361 faces receiving exactly 0 W/m^2. Provided ONLY to
    //       reproduce published runs; it re-introduces the defect.
    enum class ExchangeSelection { All, Shadow };
    ExchangeSelection exchange_selection = ExchangeSelection::All;

    // Min fraction of time in shadow for a face to keep its indirect flux.
    // Consulted ONLY when exchange_selection == Shadow. Before PTM-99 this was
    // hard-coded here at 0.3 because the config key was parsed and then never
    // bridged into this struct; it is now plumbed through.
    double min_shadow_fraction = 0.3;

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

    // Monte Carlo ray-gather view-factor builder (mc_view_factor.cpp).
    // Selected by use_monte_carlo_vf. Scales to 4M+ meshes where H-matrix
    // build memory and sparse-VF adaptive-neighbor cap both fall over.
    bool use_monte_carlo_vf = false;
    int mc_rays_per_face = 100000;       // Production default: ~10% rel err on F_ij~1e-3.

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
    // Jacobi iterations between convergence checks on the GPU. The CPU paths
    // test every iteration (the residual is already in host memory there); the
    // GPU must pay a device->host round trip for it, so it batches. 0 = auto,
    // weighed per matrix by gpu::opencl::resolveConvergenceCheckInterval();
    // > 0 pins it. This value reached the GPU solver for the first time on
    // 2026-08-15 -- before that the solver's own hard-coded 10 was the only
    // value that existed, and a 509k solve that had stagnated at iteration 2
    // still ran to 10, making GPU radiosity 1.74x slower than the CPU.
    int convergence_check_interval = 0;

    // Band-separated (visible + thermal-IR) radiosity (PTM-07 fix). Production
    // DEFAULT ON (D2, 2026-07-12): the visible band scatters/absorbs the per-face
    // Bond albedo A_i (sourced from the surface albedo config) and the IR band
    // keeps 1-eps/eps; their two absorbed indirect fluxes are summed. Requires a
    // per-face albedo (setSolarAlbedo) — without one, enableTwoBand() warns LOUD and
    // stays gray, so the default is safe. The legacy single gray-band solve is
    // recovered EXACTLY (byte-identical at A=1-eps) by setting band_separated=false.
    // The env PTM_RAD_TWO_BAND remains an equivalent activation path.
    // See docs/two-band-radiosity-derivation.tex.
    bool band_separated = true;

    // Multi-scattering control
    int max_bounces = 1;                 // Number of scattering bounces (1 = single)

    // Performance options
    bool parallel_view_factors = true;   // Parallelize view factor computation
    bool cache_view_factors = true;      // Cache view factors between timesteps
    int visibility_batch_size = 1024;    // Batch size for visibility tests

    // GPU acceleration options
    bool use_gpu = false;                // Use GPU for radiosity solver (requires OpenCL)

    // Multi-GPU radiosity dispatch (enabled only when use_gpu = true).
    // Mirrors the thermal pool's dispatch model: per-timestep solves are
    // mutually independent, so the batch can be split across N GPUs.
    int gpu_devices = 1;                 // 1 = legacy single-GPU; >1 = pool
    std::vector<int> gpu_device_ids;     // Explicit device indices (overrides gpu_devices)
    // Timesteps per pool dispatch. 0 = auto = the whole sweep in one dispatch,
    // capped by the host batch-memory budget and floored at 4 * gpu_devices;
    // see radiation::resolvePoolBatchSize().
    int gpu_batch_size = 0;
    int gpu_inter_batch_pause_ms = 0;    // Watchdog mitigation, mirrors thermal pool

    // H-matrix compression options (for large meshes)
    bool use_hmatrix = false;            // Use hierarchical matrix compression
    // Bypass the n_faces > 1M auto-disable so H-matrix + OOC is attempted regardless.
    // Intended for the probe run that determines whether the >1M disable is still
    // warranted; left off by default so existing behavior is unchanged.
    bool force_hmatrix = false;
    double hmatrix_tolerance = 1e-2;     // SVD truncation tolerance for low-rank blocks
    int hmatrix_max_depth = 10;          // Maximum quadtree depth
    int hmatrix_min_block_size = 64;     // Minimum block size before treating as leaf
    int hmatrix_max_rank = 50;           // Maximum rank for low-rank blocks

    // ACA (Adaptive Cross Approximation) options - much faster than SVD for large meshes
    bool use_aca = true;                 // Use ACA by default (50-100x faster than SVD)
    double aca_tolerance = 1e-3;         // ACA relative tolerance for convergence
    int aca_max_rank = 50;               // Maximum rank before falling back to sparse

    // H-matrix: bisect + re-ACA large admissible blocks instead of zeroing them
    // (recovers crater floor<->wall IR coupling lost above ~100k faces). Default ON.
    // Env PTM_HMATRIX_SUBDIVIDE (1/0) overrides this at build time.
    bool hmatrix_subdivide_admissible = true;

    // Anisotropic (roughness-beamed) thermal emission (Smith 1967).
    // Sun-facing rough micro-facets beam thermal IR toward the Sun; for polar
    // craters this enhances wall->floor IR (sun-dependent, amplifies the seasonal
    // cycle). rms_slope_deg = 0 (default) => pure Lambertian, byte-identical.
    double rms_slope_deg = 0.0;          // RMS micro-slope [deg]; 0 = OFF (Lambertian)
    double beam_vf_threshold = 1e-7;     // Min renormalized VF to keep a beam pair

    // Radiance-weighted view-factor pruning epsilon [W/m^2]. Applied at VF
    // ASSEMBLY time (assemble_vf_haworth.py); the sim does NOT prune -- it records
    // the configured value for provenance/logging only. 0.0 (default) = OFF,
    // byte-identical. DISTINCT from visibility_threshold (a build-time storage
    // floor). See docs/radiation-exchange.md 10.9.
    double radiance_prune_eps = 0.0;

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
