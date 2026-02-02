#include "radiation_exchange.h"
#include "../thermal/thermal_model.h"  // For ProgressBar

#include <embree4/rtcore.h>
#include <embree4/rtcore_ray.h>

#ifdef PTM_GPU_OPENCL
#include "../gpu/radiosity_opencl.h"
#endif

#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <atomic>
#include <omp.h>

namespace radiation {

// ============================================================================
// RadiositySolver Implementation
// ============================================================================

void RadiositySolver::initCommon(const std::vector<double>& emissivity,
                                  const std::vector<double>& areas) {
    emissivity_ = emissivity;
    areas_ = areas;

    size_t n = emissivity.size();
    reflectivity_.resize(n);
    for (size_t i = 0; i < n; ++i) {
        reflectivity_[i] = 1.0 - emissivity_[i];
    }

    // Pre-allocate work arrays
    radiosity_.resize(n, 0.0);
    radiosity_new_.resize(n, 0.0);
    temp_vec_.resize(n, 0.0);
}

RadiositySolver::RadiositySolver(const ViewFactorMatrix& F,
                                 const std::vector<double>& emissivity,
                                 const std::vector<double>& areas)
    : F_op_(std::make_unique<ViewFactorOperator>(F)),
      F_ptr_(&F),
      use_hmatrix_(false) {
    initCommon(emissivity, areas);
}

RadiositySolver::RadiositySolver(const HMatrixViewFactor& H,
                                 const std::vector<double>& emissivity,
                                 const std::vector<double>& areas)
    : F_op_(std::make_unique<HMatrixOperator>(H)),
      F_ptr_(nullptr),
      use_hmatrix_(true) {
    initCommon(emissivity, areas);
}

void RadiositySolver::initGPU() {
#ifdef PTM_GPU_OPENCL
    if (config_.verbose) {
        std::cout << "[RadiositySolver] initGPU called: use_gpu=" << config_.use_gpu
                  << ", use_hmatrix=" << use_hmatrix_
                  << ", F_ptr_=" << (F_ptr_ ? "valid" : "null") << std::endl;
    }

    if (!config_.use_gpu) {
        if (config_.verbose) {
            std::cout << "[RadiositySolver] GPU not requested, using CPU path" << std::endl;
        }
        return;
    }

    // GPU solver works best with sparse view factors, not H-matrix
    if (use_hmatrix_) {
        std::cerr << "Warning: GPU radiosity solver not supported with H-matrix; "
                  << "using CPU path\n";
        return;
    }

    if (!F_ptr_) {
        std::cerr << "Warning: GPU radiosity solver requires sparse view factor matrix\n";
        return;
    }

    size_t n = emissivity_.size();
    size_t nnz = F_ptr_->numNonZeros();

    // Hard limit: GPU SpMV with very large matrices is unreliable on AMD drivers
    // Even if memory fits, kernel execution can crash with >100M non-zeros
    const size_t MAX_GPU_NNZ = 100000000;  // 100M non-zeros max
    if (nnz > MAX_GPU_NNZ) {
        std::cout << "Note: Sparse matrix too large for reliable GPU radiosity solver\n";
        std::cout << "  Non-zeros: " << nnz / 1000000 << "M (limit: " << MAX_GPU_NNZ / 1000000 << "M)\n";
        std::cout << "  Using CPU radiosity solver instead (this is normal for large meshes)\n";
        return;
    }

    // Check if sparse matrix fits in GPU memory before attempting
    // CSR format requires: nnz * sizeof(float) for values
    //                    + nnz * sizeof(int) for column indices
    //                    + (n+1) * sizeof(int) for row pointers
    // Plus working buffers: 7 * n * sizeof(float) for vectors
    size_t csr_memory = nnz * sizeof(float) + nnz * sizeof(int) + (n + 1) * sizeof(int);
    size_t working_memory = 7 * n * sizeof(float);  // radiosity vectors, temps, etc.
    size_t total_gpu_memory_needed = csr_memory + working_memory;

    // Create temporary solver just to check GPU memory
    gpu::opencl::RadiosityOpenCLConfig gpu_config;
    gpu_config.max_iterations = config_.max_iterations;
    gpu_config.convergence_tolerance = config_.convergence_tolerance;
    gpu_config.verbose = config_.verbose;

    gpu_solver_ = std::make_unique<gpu::opencl::RadiosityOpenCLSolver>(n, gpu_config);

    if (!gpu_solver_->initialize()) {
        std::cerr << "Warning: Failed to initialize GPU radiosity solver; "
                  << "falling back to CPU\n";
        gpu_solver_.reset();
        return;
    }

    // Check GPU memory availability (use 50% threshold to leave headroom for kernel execution)
    // Note: 70% was too aggressive - kernels need additional working memory beyond buffers
    size_t gpu_mem_available = gpu_solver_->getDeviceInfo().global_mem_size;
    size_t gpu_mem_limit = static_cast<size_t>(gpu_mem_available * 0.50);

    if (total_gpu_memory_needed > gpu_mem_limit) {
        std::cout << "Note: Sparse matrix too large for GPU radiosity solver\n";
        std::cout << "  Required GPU memory: " << total_gpu_memory_needed / (1024*1024) << " MB\n";
        std::cout << "  GPU memory limit (50%): " << gpu_mem_limit / (1024*1024) << " MB\n";
        std::cout << "  GPU total memory: " << gpu_mem_available / (1024*1024) << " MB\n";
        std::cout << "  Using CPU radiosity solver instead (this is normal for large meshes)\n";
        gpu_solver_.reset();
        return;
    }

    if (config_.verbose) {
        std::cout << "GPU radiosity memory check passed:\n";
        std::cout << "  CSR matrix: " << csr_memory / (1024*1024) << " MB\n";
        std::cout << "  Working buffers: " << working_memory / (1024*1024) << " MB\n";
        std::cout << "  Total required: " << total_gpu_memory_needed / (1024*1024) << " MB\n";
        std::cout << "  GPU available: " << gpu_mem_available / (1024*1024) << " MB\n";
    }

    // Upload view factor CSR data
    gpu_solver_->uploadViewFactors(
        F_ptr_->values(),
        F_ptr_->colIndices(),
        F_ptr_->rowPointers());

    // Upload face properties (convert to float)
    std::vector<float> emissivity_f(n);
    std::vector<float> reflectivity_f(n);
    for (size_t i = 0; i < n; ++i) {
        emissivity_f[i] = static_cast<float>(emissivity_[i]);
        reflectivity_f[i] = static_cast<float>(reflectivity_[i]);
    }
    gpu_solver_->uploadFaceProperties(emissivity_f, reflectivity_f);

    use_gpu_ = true;

    if (config_.verbose) {
        std::cout << "GPU radiosity solver enabled\n";
    }
#endif
}

#ifdef PTM_GPU_OPENCL
std::vector<double> RadiositySolver::solveGPU(
    const std::vector<double>& temperatures,
    const std::vector<double>& solar_flux)
{
    size_t n = temperatures.size();

    // Convert to float
    std::vector<float> temps_f(n);
    std::vector<float> flux_f(n);
    for (size_t i = 0; i < n; ++i) {
        temps_f[i] = static_cast<float>(temperatures[i]);
        flux_f[i] = static_cast<float>(solar_flux[i]);
    }

    // Solve on GPU
    std::vector<float> indirect_f;
    gpu_solver_->solve(temps_f, flux_f, indirect_f);

    // Get convergence info
    last_iterations_ = gpu_solver_->getLastIterations();
    last_residual_ = gpu_solver_->getLastResidual();
    converged_ = gpu_solver_->didConverge();

    // Convert back to double
    std::vector<double> indirect_flux(n);
    for (size_t i = 0; i < n; ++i) {
        indirect_flux[i] = static_cast<double>(indirect_f[i]);
    }

    return indirect_flux;
}
#endif

std::vector<double> RadiositySolver::solve(const std::vector<double>& temperatures,
                                            const std::vector<double>& solar_flux) {
#ifdef PTM_GPU_OPENCL
    // Use GPU path if available
    if (use_gpu_ && gpu_solver_) {
        return solveGPU(temperatures, solar_flux);
    }
#endif

    size_t n = temperatures.size();

    // Compute thermal emission: E_i = ε_i σ T_i^4
    auto emission = computeThermalEmission(temperatures);

    // Compute reflected solar source: ρ_i Q_solar_i
    std::vector<double> solar_source(n);
    for (size_t i = 0; i < n; ++i) {
        solar_source[i] = reflectivity_[i] * solar_flux[i];
    }

    // Solve radiosity equation: B = E + ρ F B + ρ Q_solar
    // Rearranged: (I - ρ F) B = E + ρ Q_solar
    std::vector<double> radiosity;
    // Use Jacobi for H-matrix (no row access), Gauss-Seidel for sparse
    if (config_.use_acceleration && !use_hmatrix_) {
        radiosity = solveGaussSeidel(emission, solar_source);
    } else {
        radiosity = solveJacobi(emission, solar_source);
    }

    // Compute absorbed indirect flux at each face
    // Q_indirect_i = ε_i * Σ_j F_ij * B_j
    std::vector<double> indirect_flux(n, 0.0);
    std::vector<double> incoming(n, 0.0);

    // incoming_i = Σ_j F_ij B_j
    F_op_->matvec(radiosity, incoming);

    for (size_t i = 0; i < n; ++i) {
        // Absorbed flux = emissivity * incoming radiosity
        // (by Kirchhoff's law, absorptivity = emissivity)
        indirect_flux[i] = emissivity_[i] * incoming[i];
    }

    return indirect_flux;
}

std::vector<double> RadiositySolver::computeThermalEmission(
    const std::vector<double>& temperatures) const {

    size_t n = temperatures.size();
    std::vector<double> emission(n);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        double T = temperatures[i];
        double T2 = T * T;
        double T4 = T2 * T2;
        emission[i] = emissivity_[i] * SIGMA_SB * T4;
    }

    return emission;
}

std::vector<double> RadiositySolver::computeScatteredSolar(
    const std::vector<double>& solar_flux) const {

    size_t n = solar_flux.size();

    // First bounce: ρ_i Q_solar_i leaves face i
    std::vector<double> scattered_source(n);
    for (size_t i = 0; i < n; ++i) {
        scattered_source[i] = reflectivity_[i] * solar_flux[i];
    }

    // Received scattered solar: F^T * (ρ Q_solar)
    std::vector<double> received(n, 0.0);
    F_op_->matvec(scattered_source, received);

    // Absorbed scattered solar
    std::vector<double> absorbed(n);
    for (size_t i = 0; i < n; ++i) {
        absorbed[i] = emissivity_[i] * received[i];
    }

    return absorbed;
}

std::vector<double> RadiositySolver::solveJacobi(const std::vector<double>& emission,
                                                  const std::vector<double>& solar_source) {
    size_t n = emission.size();

    // Use local arrays instead of member arrays for thread safety
    // This allows multiple threads to call solveJacobi concurrently
    std::vector<double> radiosity(n);
    std::vector<double> radiosity_new(n);
    std::vector<double> temp_vec(n);

    // Initialize radiosity with emission + solar source
    for (size_t i = 0; i < n; ++i) {
        radiosity[i] = emission[i] + solar_source[i];
    }

    double local_residual = std::numeric_limits<double>::max();
    bool local_converged = false;
    int local_iterations = 0;

    for (int iter = 0; iter < config_.max_iterations; ++iter) {
        // B_new = E + ρ Q_solar + ρ F B
        // Note: matvec is internally parallelized over H-matrix blocks
        F_op_->matvec(radiosity, temp_vec);

        double max_change = 0.0;
        double max_val = 0.0;

        // This loop is fast, no need to parallelize (would conflict with outer parallelism)
        for (size_t i = 0; i < n; ++i) {
            radiosity_new[i] = emission[i] + solar_source[i] + reflectivity_[i] * temp_vec[i];

            double change = std::abs(radiosity_new[i] - radiosity[i]);
            max_change = std::max(max_change, change);
            max_val = std::max(max_val, std::abs(radiosity_new[i]));
        }

        // Swap buffers
        std::swap(radiosity, radiosity_new);

        // Check convergence
        double rel_residual = (max_val > 1e-10) ? max_change / max_val : max_change;
        local_residual = rel_residual;
        local_iterations = iter + 1;

        if (rel_residual < config_.convergence_tolerance) {
            local_converged = true;
            break;
        }
    }

    // Update member variables (for diagnostics) - not thread-safe but only for reporting
    last_residual_ = local_residual;
    last_iterations_ = local_iterations;
    converged_ = local_converged;

    return radiosity;
}

std::vector<double> RadiositySolver::solveGaussSeidel(const std::vector<double>& emission,
                                                       const std::vector<double>& solar_source) {
    size_t n = emission.size();

    // Gauss-Seidel requires row access; fall back to Jacobi if not available
    if (!F_ptr_) {
        return solveJacobi(emission, solar_source);
    }

    // Initialize radiosity with emission + solar source
    radiosity_.assign(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        radiosity_[i] = emission[i] + solar_source[i];
    }

    last_residual_ = std::numeric_limits<double>::max();
    converged_ = false;

    std::vector<int> cols;
    std::vector<double> vals;

    for (int iter = 0; iter < config_.max_iterations; ++iter) {
        double max_change = 0.0;
        double max_val = 0.0;

        // Gauss-Seidel: update each row using latest values
        for (size_t i = 0; i < n; ++i) {
            // Get row i of view factor matrix
            F_ptr_->getRow(static_cast<int>(i), cols, vals);

            // Compute sum F_ij * B_j using latest B values
            double sum = 0.0;
            for (size_t k = 0; k < cols.size(); ++k) {
                sum += vals[k] * radiosity_[cols[k]];
            }

            // Update B_i
            double B_new = emission[i] + solar_source[i] + reflectivity_[i] * sum;

            double change = std::abs(B_new - radiosity_[i]);
            max_change = std::max(max_change, change);
            max_val = std::max(max_val, std::abs(B_new));

            radiosity_[i] = B_new;
        }

        // Check convergence
        double rel_residual = (max_val > 1e-10) ? max_change / max_val : max_change;
        last_residual_ = rel_residual;
        last_iterations_ = iter + 1;

        if (rel_residual < config_.convergence_tolerance) {
            converged_ = true;
            break;
        }
    }

    return radiosity_;
}

std::array<double, 3> RadiositySolver::checkEnergyBalance(
    const std::vector<double>& temperatures,
    const std::vector<double>& solar_flux,
    const std::vector<double>& indirect_flux) const {

    size_t n = temperatures.size();

    double total_emission = 0.0;
    double total_solar_absorbed = 0.0;
    double total_indirect = 0.0;

    for (size_t i = 0; i < n; ++i) {
        double T = temperatures[i];
        double T4 = T * T * T * T;
        total_emission += emissivity_[i] * SIGMA_SB * T4 * areas_[i];
        total_solar_absorbed += emissivity_[i] * solar_flux[i] * areas_[i];
        total_indirect += indirect_flux[i] * areas_[i];
    }

    double imbalance = std::abs(total_emission - total_solar_absorbed - total_indirect);

    return {{total_emission, total_solar_absorbed + total_indirect, imbalance}};
}

// ============================================================================
// FaceSelector Implementation
// ============================================================================

FaceSelector::FaceSelector(const std::shared_ptr<ptm::Mesh>& mesh,
                           const RadiationConfig& config)
    : mesh_(mesh), config_(config) {

    // Use pre-computed face geometry from mesh if available
    // PTM3D's MeshUtils ensures normals point outward (away from body center)
    size_t n = mesh_->faces.size();
    face_normals_.resize(n);
    face_centers_.resize(n);

    bool use_mesh_data = (mesh_->face_centers.size() == n &&
                          mesh_->face_normals.size() == n);

    if (use_mesh_data) {
        // Use pre-computed outward-oriented normals from mesh
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            face_centers_[i] = mesh_->face_centers[i];
            face_normals_[i] = mesh_->face_normals[i];
        }
    } else {
        // Compute from scratch and ensure outward orientation
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            const auto& face = mesh_->faces[i];
            const auto& v0 = mesh_->vertices[face[0]];
            const auto& v1 = mesh_->vertices[face[1]];
            const auto& v2 = mesh_->vertices[face[2]];

            // Center
            face_centers_[i] = {{
                (v0[0] + v1[0] + v2[0]) / 3.0,
                (v0[1] + v1[1] + v2[1]) / 3.0,
                (v0[2] + v1[2] + v2[2]) / 3.0
            }};

            // Normal (cross product of edges)
            std::array<double, 3> e1 = {{v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]}};
            std::array<double, 3> e2 = {{v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]}};

            face_normals_[i] = {{
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0]
            }};

            double len = std::sqrt(face_normals_[i][0] * face_normals_[i][0] +
                                   face_normals_[i][1] * face_normals_[i][1] +
                                   face_normals_[i][2] * face_normals_[i][2]);
            if (len > 1e-10) {
                face_normals_[i][0] /= len;
                face_normals_[i][1] /= len;
                face_normals_[i][2] /= len;
            }

            // Ensure outward-pointing normals (away from body center at origin)
            // PTM3D convention: normal · center > 0 means outward
            const auto& c = face_centers_[i];
            double radial_dot = face_normals_[i][0] * c[0] +
                                face_normals_[i][1] * c[1] +
                                face_normals_[i][2] * c[2];
            if (radial_dot < 0) {
                face_normals_[i][0] = -face_normals_[i][0];
                face_normals_[i][1] = -face_normals_[i][1];
                face_normals_[i][2] = -face_normals_[i][2];
            }
        }
    }
}

FaceClassification FaceSelector::classify(
    const std::vector<std::vector<double>>& illumination_series,
    const std::vector<std::vector<double>>& temperatures) {

    size_t n = mesh_->faces.size();
    FaceClassification result;
    result.resize(n);

    // Always show progress messages for these potentially slow operations
    // Start on new line since caller may have printed without newline
    std::cout << std::endl;
    std::cout << "  Computing shadow fractions..." << std::flush;

    // Compute shadow fractions
    result.shadow_fractions = computeShadowFractions(illumination_series);

    std::cout << " done" << std::endl;

    // Extract minimum temperatures if provided
    if (!temperatures.empty() && temperatures.size() == n) {
        for (size_t i = 0; i < n; ++i) {
            if (!temperatures[i].empty()) {
                result.min_temperatures[i] = *std::min_element(
                    temperatures[i].begin(), temperatures[i].end());
            }
        }
    }

    // Classify faces based on shadow fraction
    // Note: View factors are computed for ALL faces regardless of classification.
    // The actual filtering for thermal re-runs happens based on computed indirect flux.
    for (size_t i = 0; i < n; ++i) {
        result.needs_exchange[i] = (result.shadow_fractions[i] >= config_.min_shadow_fraction);
    }

    result.finalize();

    if (config_.verbose) {
        std::cout << "Face classification: " << result.exchange_faces << " / "
                  << result.total_faces << " faces need radiation exchange ("
                  << (100.0 * result.exchange_faces / result.total_faces) << "%)"
                  << std::endl;
        std::cout << "  Mean shadow fraction: " << result.mean_shadow_fraction << std::endl;
    }

    return result;
}

FaceClassification FaceSelector::classifyGeometric() {
    // Quick classification based on geometry only
    std::vector<std::vector<double>> empty_illum;
    return classify(empty_illum);
}

std::vector<double> FaceSelector::computeShadowFractions(
    const std::vector<std::vector<double>>& illumination_series) const {

    size_t n = mesh_->faces.size();
    std::vector<double> fractions(n, 0.0);

    if (illumination_series.empty()) return fractions;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        if (i < illumination_series.size() && !illumination_series[i].empty()) {
            int shadow_count = 0;
            for (double flux : illumination_series[i]) {
                if (flux < 1e-6) {  // Effectively zero flux = in shadow
                    shadow_count++;
                }
            }
            fractions[i] = static_cast<double>(shadow_count) / illumination_series[i].size();
        }
    }

    return fractions;
}

// ============================================================================
// RadiationExchangeManager Implementation
// ============================================================================

RadiationExchangeManager::RadiationExchangeManager(
    const std::shared_ptr<ptm::Mesh>& mesh,
    const RadiationConfig& config)
    : mesh_(mesh), config_(config) {

    // Automatic H-matrix selection for large meshes
    // Dense view factors require O(n^2) memory which is impractical for large meshes
    // Threshold: 100K faces -> 10B entries -> 80GB for dense matrix (infeasible)
    const size_t HMATRIX_AUTO_THRESHOLD = 100000;
    size_t n_faces = mesh_->faces.size();

    if (n_faces > HMATRIX_AUTO_THRESHOLD && !config_.use_hmatrix) {
        std::cout << "Large mesh detected (" << n_faces << " faces)" << std::endl;
        std::cout << "  Automatically enabling H-matrix compression (dense would require ~"
                  << (n_faces * n_faces * 8ULL) / (1024ULL * 1024 * 1024) << " GB)" << std::endl;
        config_.use_hmatrix = true;
    }

    if (config_.verbose) {
        std::cout << "[RadiationExchangeManager] Config received:" << std::endl;
        std::cout << "  use_hmatrix=" << (config_.use_hmatrix ? "true" : "false") << std::endl;
        std::cout << "  use_gpu=" << (config_.use_gpu ? "true" : "false") << std::endl;
        std::cout << std::scientific << "  visibility_threshold=" << config_.visibility_threshold
                  << std::defaultfloat << std::endl;
        if (config_.use_hmatrix) {
            std::cout << "  hmatrix_tolerance=" << config_.hmatrix_tolerance << std::endl;
            std::cout << "  hmatrix_max_rank=" << config_.hmatrix_max_rank << std::endl;
        }
    }

    size_t n = mesh_->faces.size();
    emissivity_.assign(n, 0.95);  // Default emissivity
    areas_.resize(n);

    // Compute face areas
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const auto& face = mesh_->faces[i];
        const auto& v0 = mesh_->vertices[face[0]];
        const auto& v1 = mesh_->vertices[face[1]];
        const auto& v2 = mesh_->vertices[face[2]];

        std::array<double, 3> e1 = {{v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]}};
        std::array<double, 3> e2 = {{v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]}};
        std::array<double, 3> cross = {{
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0]
        }};
        areas_[i] = 0.5 * std::sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);
    }

    indirect_flux_.resize(n, 0.0);
}

RadiationExchangeManager::~RadiationExchangeManager() {
    cleanup();
}

void RadiationExchangeManager::cleanup() {
    if (scene_) {
        rtcReleaseScene(static_cast<RTCScene>(scene_));
        scene_ = nullptr;
    }
    if (device_) {
        rtcReleaseDevice(static_cast<RTCDevice>(device_));
        device_ = nullptr;
    }
    scene_initialized_ = false;
}

bool RadiationExchangeManager::initializeScene() {
    if (scene_initialized_) return true;

    // Create Embree device
    device_ = rtcNewDevice(nullptr);
    if (!device_) {
        std::cerr << "Error: Failed to create Embree device" << std::endl;
        return false;
    }

    // Create scene
    scene_ = rtcNewScene(static_cast<RTCDevice>(device_));
    if (!scene_) {
        std::cerr << "Error: Failed to create Embree scene" << std::endl;
        cleanup();
        return false;
    }

    RTCScene rtc_scene = static_cast<RTCScene>(scene_);
    RTCDevice rtc_device = static_cast<RTCDevice>(device_);

    // Create geometry
    RTCGeometry geom = rtcNewGeometry(rtc_device, RTC_GEOMETRY_TYPE_TRIANGLE);

    // Set vertices
    size_t n_verts = mesh_->vertices.size();
    float* vertices = static_cast<float*>(
        rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0,
                                RTC_FORMAT_FLOAT3, 3 * sizeof(float), n_verts));

    for (size_t i = 0; i < n_verts; ++i) {
        vertices[3 * i + 0] = static_cast<float>(mesh_->vertices[i][0]);
        vertices[3 * i + 1] = static_cast<float>(mesh_->vertices[i][1]);
        vertices[3 * i + 2] = static_cast<float>(mesh_->vertices[i][2]);
    }

    // Set indices
    size_t n_faces = mesh_->faces.size();
    unsigned int* indices = static_cast<unsigned int*>(
        rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0,
                                RTC_FORMAT_UINT3, 3 * sizeof(unsigned int), n_faces));

    for (size_t i = 0; i < n_faces; ++i) {
        indices[3 * i + 0] = static_cast<unsigned int>(mesh_->faces[i][0]);
        indices[3 * i + 1] = static_cast<unsigned int>(mesh_->faces[i][1]);
        indices[3 * i + 2] = static_cast<unsigned int>(mesh_->faces[i][2]);
    }

    rtcCommitGeometry(geom);
    rtcAttachGeometry(rtc_scene, geom);
    rtcReleaseGeometry(geom);
    rtcCommitScene(rtc_scene);

    // Check for Embree errors after scene commit
    // Large meshes can fail during BVH construction
    RTCError err = rtcGetDeviceError(static_cast<RTCDevice>(device_));
    if (err != RTC_ERROR_NONE) {
        std::cerr << "Embree scene commit error (code " << err << ")" << std::endl;
        std::cerr << "Mesh may be too large (" << mesh_->faces.size() << " faces)" << std::endl;
        cleanup();
        return false;
    }

    scene_initialized_ = true;
    return true;
}

bool RadiationExchangeManager::computeViewFactorsOnly() {
    // Compute view factors without face classification or solver setup.
    // This is used for early computation before illumination data is available.
    // The view factors can be saved to disk and loaded later during prepare().

    if (!config_.enabled) {
        return false;
    }

    size_t n_faces = mesh_->faces.size();
    std::cout << "Computing view factors for " << n_faces << " faces..." << std::endl;

    if (n_faces > 1000000) {
        size_t embree_estimate = n_faces * 80;
        std::cout << "  Estimated Embree BVH: ~" << embree_estimate / (1024*1024) << " MB" << std::endl;
    }

    // Initialize Embree scene
    if (!initializeScene()) {
        std::cerr << "Error: Failed to initialize scene for view factor computation" << std::endl;
        return false;
    }

    // Choose between H-matrix (compressed) and standard sparse view factors
    // AUTO-DISABLE H-matrix for very large meshes (>1M faces) - falls back to sparse
    // This is because H-matrix working memory during build exceeds available RAM
    // even with out-of-core storage and index clearing optimizations.
    bool use_hmatrix = config_.use_hmatrix;
    std::cout << "  [DEBUG] View factor method selection:" << std::endl;
    std::cout << "    config_.use_hmatrix = " << (config_.use_hmatrix ? "true" : "false") << std::endl;
    std::cout << "    n_faces = " << n_faces << std::endl;
    std::cout << "    auto-disable threshold = 1,000,000 faces" << std::endl;

    // Aggressive threshold: disable H-matrix for meshes > 1M faces
    // H-matrix build memory scales as O(N²/B) where B is block size
    // For 2M faces with typical block sizes, this exceeds 384 GB RAM
    if (use_hmatrix && n_faces > 1000000) {
        std::cout << std::endl;
        std::cout << "  *** H-MATRIX AUTO-DISABLED ***" << std::endl;
        std::cout << "  Mesh has " << n_faces << " faces (> 1M threshold)" << std::endl;
        std::cout << "  H-matrix build memory exceeds available RAM for meshes this large." << std::endl;
        std::cout << "  Falling back to sparse view factors with adaptive neighbor selection." << std::endl;
        std::cout << std::endl;
        use_hmatrix = false;

        // Enable adaptive neighbor selection for memory-efficient sparse matrix
        // Each face gets as many neighbors as needed for 99% captured energy (configurable)
        config_.adaptive_neighbors = true;
        std::cout << "  Adaptive neighbor selection enabled (target: "
                  << (config_.captured_energy_target * 100) << "% captured energy)" << std::endl;
    }

    if (use_hmatrix) {
        std::cout << "Using H-matrix compression for view factors..." << std::endl;

        HMatrixConfig hconfig;
        hconfig.compression_tolerance = config_.hmatrix_tolerance;
        hconfig.max_tree_depth = config_.hmatrix_max_depth;
        hconfig.min_block_size = config_.hmatrix_min_block_size;
        hconfig.max_svd_rank = config_.hmatrix_max_rank;
        hconfig.verbose = config_.verbose;

        // ACA configuration (Adaptive Cross Approximation - much faster)
        hconfig.compression_method = config_.use_aca ? CompressionMethod::ACA : CompressionMethod::SVD;
        hconfig.aca_tolerance = config_.aca_tolerance;
        hconfig.aca_max_rank = config_.aca_max_rank;
        hconfig.hmatrix_threads = config_.hmatrix_threads;

        if (config_.use_aca) {
            std::cout << "  Using ACA compression (tolerance=" << config_.aca_tolerance
                      << ", max_rank=" << config_.aca_max_rank << ")" << std::endl;
        } else {
            std::cout << "  Using SVD compression" << std::endl;
        }

        hmatrix_view_factors_ = std::make_unique<HMatrixViewFactor>();
        hmatrix_view_factors_->build(mesh_,
                                     static_cast<RTCDevice>(device_),
                                     static_cast<RTCScene>(scene_),
                                     config_,
                                     hconfig);

        using_hmatrix_ = true;

        std::cout << "H-matrix view factors computed:" << std::endl;
        std::cout << "  Memory usage: " << hmatrix_view_factors_->memoryBytes() / (1024.0 * 1024.0)
                  << " MB" << std::endl;
        std::cout << "  Compression ratio: " << hmatrix_view_factors_->compressionRatio() << "x" << std::endl;
        if (config_.verbose) {
            hmatrix_view_factors_->printStatistics();
        }
    } else {
        std::cout << "Using sparse view factor matrix..." << std::endl;

        ViewFactorCalculator calculator(mesh_,
                                        static_cast<RTCDevice>(device_),
                                        static_cast<RTCScene>(scene_),
                                        config_);

        view_factors_ = calculator.computeAll();
        using_hmatrix_ = false;

        std::cout << "Sparse view factors computed:" << std::endl;
        std::cout << "  Non-zeros: " << view_factors_.numNonZeros() << std::endl;
        std::cout << "  Memory usage: " << view_factors_.memoryUsage() / (1024.0 * 1024.0)
                  << " MB" << std::endl;
    }

    // Note: We don't set view_factors_loaded_ = true here because we want
    // prepare() to recognize these as freshly computed (for saving to disk).
    // The solver is NOT created here - that happens in prepare() when
    // illumination data and face classification are available.

    return true;
}

bool RadiationExchangeManager::prepare(
    const std::vector<std::vector<double>>& illumination_series) {

    if (!config_.enabled) {
        is_active_ = false;
        return false;
    }

    // Note: Auto H-matrix selection is done in constructor

    size_t n_faces = mesh_->faces.size();
    if (n_faces > 1000000) {
        // Additional info for very large meshes
        size_t embree_estimate = n_faces * 80;  // ~80 bytes per face for BVH
        std::cout << "  Estimated Embree BVH: ~" << embree_estimate / (1024*1024) << " MB" << std::endl;
    }

    // Initialize Embree scene
    if (!initializeScene()) {
        std::cerr << "Error: Failed to initialize scene for radiation exchange" << std::endl;
        is_active_ = false;
        return false;
    }

    // Classify faces
    FaceSelector selector(mesh_, config_);
    classification_ = selector.classify(illumination_series);

    if (classification_.exchange_faces == 0) {
        if (config_.verbose) {
            std::cout << "No faces require radiation exchange" << std::endl;
        }
        is_active_ = false;
        return false;
    }

    // Check if view factors were pre-loaded from file
    if (view_factors_loaded_) {
        if (config_.verbose) {
            std::cout << "Using pre-loaded view factors (skipping computation)" << std::endl;
        }

        // Create radiosity solver with the loaded view factors
        if (using_hmatrix_) {
            solver_ = std::make_unique<RadiositySolver>(*hmatrix_view_factors_, emissivity_, areas_);
            if (config_.verbose) {
                std::cout << "Radiation exchange prepared (loaded H-matrix):" << std::endl;
                std::cout << "  Exchange faces: " << classification_.exchange_faces << std::endl;
                std::cout << "  Memory usage: " << hmatrix_view_factors_->memoryBytes() / (1024.0 * 1024.0)
                          << " MB" << std::endl;
            }
        } else {
            solver_ = std::make_unique<RadiositySolver>(view_factors_, emissivity_, areas_);
            if (config_.verbose) {
                std::cout << "Radiation exchange prepared (loaded sparse):" << std::endl;
                std::cout << "  Exchange faces: " << classification_.exchange_faces << std::endl;
                std::cout << "  View factor matrix size: " << view_factors_.numNonZeros()
                          << " non-zeros" << std::endl;
                std::cout << "  Memory usage: " << view_factors_.memoryUsage() / (1024.0 * 1024.0)
                          << " MB" << std::endl;
            }
        }

        // CRITICAL: Set config and initialize GPU for loaded view factors
        // Without this, GPU memory check is skipped and GPU crashes on large matrices
        solver_->setConfig(config_);
        solver_->initGPU();

        is_active_ = true;
        return true;
    }

    // Compute view factors for ALL faces, not just exchange faces.
    // This is critical because:
    // 1. Exchange faces (shadowed floor) need to RECEIVE radiation from illuminated walls
    // 2. Illuminated wall faces are NOT exchange faces (they're well-lit)
    // 3. For floor to receive radiation from wall, we need F_floor,wall which requires
    //    computing view factors FROM floors TO walls (and walls back to floors)
    // 4. The radiosity solver needs B_wall = ε*σ*T_wall^4 to propagate to floors
    //
    // Computing view factors for all faces ensures that warm illuminated surfaces
    // can contribute thermal radiation to cold shadowed surfaces.
    if (config_.verbose) {
        std::cout << "Computing view factors for ALL " << mesh_->faces.size()
                  << " faces (not just " << classification_.exchange_faces
                  << " exchange faces)..." << std::endl;
        std::cout << "  This ensures illuminated walls can radiate to shadowed floors." << std::endl;
    }

    // AUTO-DISABLE H-matrix for very large meshes (>1M faces) - falls back to sparse
    // This is because H-matrix working memory during build exceeds available RAM
    // even with out-of-core storage and index clearing optimizations.
    bool use_hmatrix = config_.use_hmatrix;
    std::cout << "  [DEBUG] View factor method selection (prepare):" << std::endl;
    std::cout << "    config_.use_hmatrix = " << (config_.use_hmatrix ? "true" : "false") << std::endl;
    std::cout << "    n_faces = " << n_faces << std::endl;
    std::cout << "    auto-disable threshold = 1,000,000 faces" << std::endl;

    if (use_hmatrix && n_faces > 1000000) {
        std::cout << std::endl;
        std::cout << "  *** H-MATRIX AUTO-DISABLED (prepare) ***" << std::endl;
        std::cout << "  Mesh has " << n_faces << " faces (> 1M threshold)" << std::endl;
        std::cout << "  H-matrix build memory exceeds available RAM for meshes this large." << std::endl;
        std::cout << "  Falling back to sparse view factors with adaptive neighbor selection." << std::endl;
        std::cout << std::endl;
        use_hmatrix = false;

        // Enable adaptive neighbor selection for memory-efficient sparse matrix
        // Each face gets as many neighbors as needed for 99% captured energy (configurable)
        config_.adaptive_neighbors = true;
        std::cout << "  Adaptive neighbor selection enabled (target: "
                  << (config_.captured_energy_target * 100) << "% captured energy)" << std::endl;
    }

    // Choose between H-matrix (compressed) and standard sparse view factors
    if (use_hmatrix) {
        std::cout << "Using H-matrix compression for view factors..." << std::endl;

        // Create H-matrix configuration
        HMatrixConfig hconfig;
        hconfig.compression_tolerance = config_.hmatrix_tolerance;
        hconfig.max_tree_depth = config_.hmatrix_max_depth;
        hconfig.min_block_size = config_.hmatrix_min_block_size;
        hconfig.max_svd_rank = config_.hmatrix_max_rank;
        hconfig.verbose = config_.verbose;

        // ACA configuration (Adaptive Cross Approximation - much faster)
        hconfig.compression_method = config_.use_aca ? CompressionMethod::ACA : CompressionMethod::SVD;
        hconfig.aca_tolerance = config_.aca_tolerance;
        hconfig.aca_max_rank = config_.aca_max_rank;
        hconfig.hmatrix_threads = config_.hmatrix_threads;

        if (config_.use_aca) {
            std::cout << "  Using ACA compression (tolerance=" << config_.aca_tolerance
                      << ", max_rank=" << config_.aca_max_rank << ")" << std::endl;
        } else {
            std::cout << "  Using SVD compression" << std::endl;
        }

        hmatrix_view_factors_ = std::make_unique<HMatrixViewFactor>();
        hmatrix_view_factors_->build(mesh_,
                                     static_cast<RTCDevice>(device_),
                                     static_cast<RTCScene>(scene_),
                                     config_,
                                     hconfig);

        // Create radiosity solver with H-matrix
        solver_ = std::make_unique<RadiositySolver>(*hmatrix_view_factors_, emissivity_, areas_);
        using_hmatrix_ = true;

        // Always show H-matrix summary (memory savings are important info)
        std::cout << "H-matrix view factors computed:" << std::endl;
        std::cout << "  Memory usage: " << hmatrix_view_factors_->memoryBytes() / (1024.0 * 1024.0)
                  << " MB" << std::endl;
        std::cout << "  Compression ratio: " << hmatrix_view_factors_->compressionRatio() << "x" << std::endl;
        if (config_.verbose) {
            hmatrix_view_factors_->printStatistics();
        }
    } else {
        // Standard sparse view factor computation
        std::cout << "Using sparse view factor matrix..." << std::endl;

        ViewFactorCalculator calculator(mesh_,
                                        static_cast<RTCDevice>(device_),
                                        static_cast<RTCScene>(scene_),
                                        config_);

        // Compute for ALL faces so that wall->floor radiation is captured
        view_factors_ = calculator.computeAll();

        // Create radiosity solver
        solver_ = std::make_unique<RadiositySolver>(view_factors_, emissivity_, areas_);
        using_hmatrix_ = false;

        // Always show sparse matrix summary
        std::cout << "Sparse view factors computed:" << std::endl;
        std::cout << "  Non-zeros: " << view_factors_.numNonZeros() << std::endl;
        std::cout << "  Memory usage: " << view_factors_.memoryUsage() / (1024.0 * 1024.0)
                  << " MB" << std::endl;
    }

    solver_->setConfig(config_);

    // Initialize GPU solver if requested (must be after setConfig)
    solver_->initGPU();

    is_active_ = true;

    return true;
}

std::vector<double> RadiationExchangeManager::computeIndirectFlux(
    const std::vector<double>& temperatures,
    const std::vector<double>& solar_flux) {

    size_t n = mesh_->faces.size();
    indirect_flux_.assign(n, 0.0);

    if (!is_active_ || !solver_) {
        return indirect_flux_;
    }

    // Solve radiosity equation
    auto flux = solver_->solve(temperatures, solar_flux);

    // Map back to all faces
    for (size_t i = 0; i < classification_.exchange_face_indices.size(); ++i) {
        int face_idx = classification_.exchange_face_indices[i];
        indirect_flux_[face_idx] = flux[face_idx];
    }

    // Update statistics
    double sum = 0.0;
    double max_flux = 0.0;
    for (int idx : classification_.exchange_face_indices) {
        sum += indirect_flux_[idx];
        max_flux = std::max(max_flux, indirect_flux_[idx]);
    }

    stats_.mean_indirect_flux = sum / std::max(size_t(1), classification_.exchange_faces);
    stats_.max_indirect_flux = max_flux;
    stats_.solver_iterations = solver_->getLastIterations();
    stats_.solver_residual = solver_->getLastResidual();

    return indirect_flux_;
}

void RadiationExchangeManager::setEmissivity(double emissivity) {
    std::fill(emissivity_.begin(), emissivity_.end(), emissivity);
}

void RadiationExchangeManager::setEmissivity(const std::vector<double>& emissivity) {
    emissivity_ = emissivity;
}

void RadiationExchangeManager::setConfig(const RadiationConfig& config) {
    config_ = config;
    if (solver_) {
        solver_->setConfig(config_);
    }
}

bool RadiationExchangeManager::loadViewFactors(const std::string& filename) {
    // Try to detect file format from magic number
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        if (config_.verbose) {
            std::cout << "View factor file not found: " << filename << std::endl;
        }
        return false;
    }

    // Read magic number to detect format
    uint32_t magic = 0;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.close();

    bool success = false;

    // H-matrix format: magic = 0x484D4154 ("HMAT")
    // Sparse format: magic = 0x5646464D ("VFFM")
    if (magic == 0x484D4154) {
        // H-matrix format
        // For large meshes, use memory-mapped loading to avoid loading entire matrix into RAM
        size_t n_faces = mesh_->faces.size();
        const size_t MMAP_THRESHOLD_FACES = 500000;  // Use mmap for meshes > 500K faces

        bool use_mmap = (n_faces >= MMAP_THRESHOLD_FACES);

        if (use_mmap) {
            std::cout << "Large mesh detected (" << n_faces << " faces)" << std::endl;
            std::cout << "  Using memory-mapped H-matrix loading to minimize RAM usage" << std::endl;
        }

        if (config_.verbose) {
            std::cout << "Loading H-matrix view factors from: " << filename
                      << (use_mmap ? " (mmap)" : " (RAM)") << std::endl;
        }

        hmatrix_view_factors_ = std::make_unique<HMatrixViewFactor>();

        if (use_mmap) {
            success = hmatrix_view_factors_->readBinaryMmap(filename);
        } else {
            success = hmatrix_view_factors_->readBinary(filename);
        }

        if (success) {
            using_hmatrix_ = true;
            config_.use_hmatrix = true;  // Ensure config reflects loaded format

            if (config_.verbose) {
                std::cout << "Loaded H-matrix view factors:" << std::endl;
                std::cout << "  Size: " << hmatrix_view_factors_->numRows() << " x "
                          << hmatrix_view_factors_->numCols() << std::endl;
                std::cout << "  Memory: " << hmatrix_view_factors_->memoryBytes() / (1024.0 * 1024.0)
                          << " MB" << (use_mmap ? " (index only, data on disk)" : "") << std::endl;
                if (use_mmap) {
                    std::cout << "  Mode: Memory-mapped (on-demand block access)" << std::endl;
                }
                hmatrix_view_factors_->printStatistics();
            }
        } else {
            hmatrix_view_factors_.reset();
            std::cerr << "Error: Failed to load H-matrix view factors from: " << filename << std::endl;
        }
    } else if (magic == 0x5646464D) {
        // Sparse format
        if (config_.verbose) {
            std::cout << "Loading sparse view factors from: " << filename << std::endl;
        }

        success = view_factors_.readBinary(filename);

        if (success) {
            using_hmatrix_ = false;
            config_.use_hmatrix = false;

            if (config_.verbose) {
                std::cout << "Loaded sparse view factors:" << std::endl;
                std::cout << "  Size: " << view_factors_.numRows() << " x "
                          << view_factors_.numCols() << std::endl;
                std::cout << "  Non-zeros: " << view_factors_.numNonZeros() << std::endl;
                std::cout << "  Memory: " << view_factors_.memoryUsage() / (1024.0 * 1024.0)
                          << " MB" << std::endl;
            }
        } else {
            std::cerr << "Error: Failed to load sparse view factors from: " << filename << std::endl;
        }
    } else {
        std::cerr << "Error: Unknown view factor file format (magic: 0x"
                  << std::hex << magic << std::dec << ")" << std::endl;
        return false;
    }

    if (success) {
        view_factors_loaded_ = true;

        // Verify matrix dimensions match mesh
        size_t n_faces = mesh_->faces.size();
        size_t matrix_size = using_hmatrix_ ? hmatrix_view_factors_->numRows() : view_factors_.numRows();

        if (matrix_size != n_faces) {
            std::cerr << "Warning: View factor matrix size (" << matrix_size
                      << ") does not match mesh face count (" << n_faces << ")" << std::endl;
            std::cerr << "  View factors will be recomputed." << std::endl;
            view_factors_loaded_ = false;
            hmatrix_view_factors_.reset();
            return false;
        }
    }

    return success;
}

} // namespace radiation
