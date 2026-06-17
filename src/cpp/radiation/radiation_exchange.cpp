#include "radiation_exchange.h"
#include "smith_beaming.h"              // Smith (1967) anisotropic emission chi table
#include "mc_view_factor.h"             // Monte Carlo view factor builder
#include "../thermal/thermal_model.h"  // For ProgressBar

#include <embree4/rtcore.h>
#include <embree4/rtcore_ray.h>

#ifdef PTM_GPU_OPENCL
#include "../gpu/radiosity_opencl.h"
#include "../gpu/radiosity_gpu_pool.h"
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

    // Optional radiation-emissivity override (PTM_RAD_EMISSIVITY) for consistency
    // studies: forces the radiosity emissivity (and hence reflectivity = 1-eps below)
    // to a single value, independent of the body/config-propagation path.
    if (const char* re = std::getenv("PTM_RAD_EMISSIVITY")) {
        double e = std::atof(re);
        if (e > 0.0 && e <= 1.0 && !emissivity_.empty()) {
            std::fill(emissivity_.begin(), emissivity_.end(), e);
            std::cout << "  [PTM_RAD_EMISSIVITY] radiosity emissivity forced to " << e
                      << " (reflectivity 1-eps=" << (1.0 - e) << ")" << std::endl;
        }
    }

    size_t n = emissivity.size();
    reflectivity_.resize(n);
    for (size_t i = 0; i < n; ++i) {
        reflectivity_[i] = 1.0 - emissivity_[i];
    }

    // Albedo fix (PTM_SOLAR_ALBEDO): the reflected-SOLAR source should scatter the
    // solar Bond albedo (~0.12), not the thermal reflectivity 1-eps (~0.05). When the
    // env is set we fill a separate solar_reflectivity_ used only for the solar source
    // (computeScatteredSolar + the solar_source term in solve); the thermal multiple-
    // scattering rho*F*B keeps 1-eps. Unset => empty => byte-identical legacy behavior.
    if (const char* sa = std::getenv("PTM_SOLAR_ALBEDO")) {
        double a = std::atof(sa);
        if (a > 0.0 && a < 1.0) {
            solar_reflectivity_.assign(n, a);
            std::cout << "  [PTM_SOLAR_ALBEDO] reflected-solar source uses Bond albedo "
                      << a << " (was 1-eps=" << (n ? reflectivity_[0] : 0.0) << ")\n";
        }
    }

    // Pre-allocate work arrays
    radiosity_.resize(n, 0.0);
    radiosity_new_.resize(n, 0.0);
    temp_vec_.resize(n, 0.0);

    // View-factor row normalization (energy conservation / radiosity stability).
    // Compute the row sums Σ_j F_ij once via a single F·1 matvec. The M1 point
    // estimator overshoots in the near field, so on fine/concave meshes some rows
    // sum to > 1, which makes the radiosity gain > 1 and the coupling diverge.
    // Store 1/max(1, Σ_j F_ij) and apply it after every F·x in the solve.
    {
        std::vector<double> ones(n, 1.0), rowsum(n, 0.0);
        if (F_op_) F_op_->matvec(ones, rowsum);
        vf_row_norm_.assign(n, 1.0);
        size_t n_over = 0;
        double max_rs = 0.0, sum_rs = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double rs = rowsum[i];
            if (rs > 1.0) { vf_row_norm_[i] = 1.0 / rs; ++n_over; }
            max_rs = std::max(max_rs, rs);
            sum_rs += rs;
        }
        std::cout << "[Radiosity] View-factor row sums Σ_j F_ij: max=" << max_rs
                  << ", mean=" << (n ? sum_rs / static_cast<double>(n) : 0.0)
                  << " (" << n_over << "/" << n
                  << " rows > 1 renormalized for energy conservation)" << std::endl;
        // Optional diagnostic: dump per-face VF row sums for completeness checks
        // (e.g. is the floor<->distant-wall coupling captured?). Set the env var
        // PTM_DUMP_VF_ROWSUMS to a path. Format: uint32 n, then n float64.
        if (const char* dp = std::getenv("PTM_DUMP_VF_ROWSUMS")) {
            std::ofstream ro(dp, std::ios::binary);
            if (ro) {
                uint32_t nn = static_cast<uint32_t>(n);
                ro.write(reinterpret_cast<const char*>(&nn), sizeof(nn));
                ro.write(reinterpret_cast<const char*>(rowsum.data()),
                         static_cast<std::streamsize>(n) * sizeof(double));
                std::cout << "[Radiosity] Dumped per-face VF row sums -> " << dp << std::endl;
            }
        }
    }
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

// Default destructor in the .cpp where forward-declared GPU types are
// fully visible.
RadiositySolver::~RadiositySolver() = default;

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

    // Soft limit: GPU SpMV with very large sparse matrices was historically
    // unreliable on AMD drivers (kernel execution could crash with >100M
    // non-zeros). The limit is bypassed when the user explicitly opts into
    // the multi-GPU pool path (gpu_devices > 1 or gpu_device_ids set) —
    // they are committing to the new code path and accept the risk for the
    // larger meshes the pool was designed to handle.
    const size_t MAX_GPU_NNZ = 100000000;  // 100M non-zeros default ceiling
    const bool want_pool = (config_.gpu_devices > 1)
                            || !config_.gpu_device_ids.empty();
    if (nnz > MAX_GPU_NNZ && !want_pool) {
        std::cout << "Note: Sparse matrix too large for reliable single-GPU radiosity solver\n";
        std::cout << "  Non-zeros: " << nnz / 1000000 << "M (limit: " << MAX_GPU_NNZ / 1000000 << "M)\n";
        std::cout << "  Using CPU radiosity solver instead (this is normal for large meshes)\n";
        std::cout << "  Tip: enable the multi-GPU pool with radiation.gpu_devices > 1 to bypass this limit.\n";
        return;
    }
    if (nnz > MAX_GPU_NNZ && want_pool) {
        std::cout << "Note: Sparse matrix has " << nnz / 1000000 << "M non-zeros (>"
                  << MAX_GPU_NNZ / 1000000
                  << "M soft limit); proceeding because multi-GPU pool was requested\n";
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

    // When the multi-GPU pool is requested, skip the single-GPU solver path
    // entirely — the pool's per-device persistent solvers will handle every
    // solve via solveBatch(). Avoids redundant device 0 allocation and
    // dodges the >100M-nnz single-GPU stability concerns the pool was
    // designed to bypass.
    std::vector<float> emissivity_f(n);
    std::vector<float> reflectivity_f(n);
    for (size_t i = 0; i < n; ++i) {
        emissivity_f[i] = static_cast<float>(emissivity_[i]);
        reflectivity_f[i] = static_cast<float>(reflectivity_[i]);
    }

    if (!want_pool) {
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
        gpu_solver_->uploadFaceProperties(emissivity_f, reflectivity_f);

        use_gpu_ = true;
    }

    // Multi-GPU dispatch pool (only when gpu_devices > 1). The single-GPU
    // path above stays in place even when the pool is active so that
    // single-timestep solve() callers (the CPU fallback hook + direct
    // computeIndirectFlux callers) still work; the batched solveBatch()
    // path is what actually uses the pool.
    if (want_pool) {
        // Resolve which physical devices to use (mirrors the thermal pool).
        std::vector<gpu::opencl::OpenCLDeviceInfo> all_gpus =
            gpu::opencl::OpenCLContext::enumerateGPUs(true);
        std::vector<gpu::opencl::OpenCLDeviceInfo> chosen;
        if (!config_.gpu_device_ids.empty()) {
            for (int id : config_.gpu_device_ids) {
                if (id >= 0 && static_cast<size_t>(id) < all_gpus.size()) {
                    chosen.push_back(all_gpus[id]);
                } else {
                    std::cerr << "Warning: rad-gpu-device-id " << id
                              << " out of range (have " << all_gpus.size()
                              << " GPUs); skipping\n";
                }
            }
        } else {
            const int n_req = (config_.gpu_devices == 0)
                ? static_cast<int>(all_gpus.size())
                : std::min<int>(config_.gpu_devices,
                                static_cast<int>(all_gpus.size()));
            for (int i = 0; i < n_req; ++i) chosen.push_back(all_gpus[i]);
        }

        if (chosen.size() < 2) {
            if (config_.verbose) {
                std::cout << "[Radiosity Pool] Pool not instantiated "
                             "(only " << chosen.size() << " device(s) selected); "
                             "single-GPU path remains active\n";
            }
        } else {
            gpu::opencl::RadiosityGPUPoolConfig pool_cfg;
            pool_cfg.devices = chosen;
            pool_cfg.inter_batch_pause_ms = config_.gpu_inter_batch_pause_ms;
            pool_cfg.verbose = config_.verbose;

            gpu::opencl::RadiosityOpenCLConfig solver_cfg;
            solver_cfg.max_iterations = config_.max_iterations;
            solver_cfg.convergence_tolerance = config_.convergence_tolerance;
            solver_cfg.verbose = config_.verbose;
            // Cross-batch warm-start has no meaning across devices — disable
            // it so each batch's first timestep starts from cold radiosity.
            // Cost: typically 5-10 extra Jacobi iterations on the first
            // timestep of each batch.
            solver_cfg.warm_start = false;

            gpu_pool_ = std::make_unique<gpu::opencl::RadiosityGPUDevicePool>(
                std::move(pool_cfg));
            if (!gpu_pool_->initialize(n, solver_cfg)) {
                std::cerr << "Warning: Failed to initialize multi-GPU radiosity "
                             "pool; falling back to single-GPU path\n";
                gpu_pool_.reset();
            } else if (!gpu_pool_->uploadStaticData(F_ptr_->values(),
                                                     F_ptr_->colIndices(),
                                                     F_ptr_->rowPointers(),
                                                     emissivity_f,
                                                     reflectivity_f)) {
                std::cerr << "Warning: Failed to replicate radiosity static "
                             "data to pool devices; falling back to "
                             "single-GPU path\n";
                gpu_pool_.reset();
            } else {
                pool_batch_size_ = (config_.gpu_batch_size > 0)
                    ? config_.gpu_batch_size
                    : 4 * static_cast<int>(gpu_pool_->numDevices());
                if (config_.verbose) {
                    std::cout << "[Radiosity Pool] Active: "
                              << gpu_pool_->numDevices() << " GPU(s), "
                              << "batch_size=" << pool_batch_size_
                              << " timesteps per dispatch\n";
                }
            }
        }
    }

    if (config_.verbose) {
        std::cout << "GPU radiosity solver enabled\n";
    }
#endif
}

bool RadiositySolver::hasGPUPool() const {
#ifdef PTM_GPU_OPENCL
    return gpu_pool_ != nullptr;
#else
    return false;
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

    // Convert back to double, applying the view-factor row normalization
    // (Σ_j F_ij ≤ 1) for energy conservation / radiosity-gain stability.
    std::vector<double> indirect_flux(n);
    for (size_t i = 0; i < n; ++i) {
        const double norm = (i < vf_row_norm_.size()) ? vf_row_norm_[i] : 1.0;
        indirect_flux[i] = static_cast<double>(indirect_f[i]) * norm;
    }

    return indirect_flux;
}
#endif

std::vector<std::vector<double>> RadiositySolver::solveBatch(
    const std::vector<std::vector<double>>& temperatures_per_timestep,
    const std::vector<std::vector<double>>& solar_flux_per_timestep)
{
    const size_t n_steps = temperatures_per_timestep.size();
    std::vector<std::vector<double>> indirect_per_timestep(n_steps);

    // Single-device or CPU fallback: serial loop over solve(). Semantically
    // identical to the caller running its own per-timestep loop.
#ifdef PTM_GPU_OPENCL
    if (!gpu_pool_) {
#endif
        for (size_t k = 0; k < n_steps; ++k) {
            indirect_per_timestep[k] =
                solve(temperatures_per_timestep[k], solar_flux_per_timestep[k]);
        }
        return indirect_per_timestep;
#ifdef PTM_GPU_OPENCL
    }

    // Multi-GPU dispatch: split the n_steps timesteps round-robin across
    // pool devices. Each device's worker iterates over its assigned slice
    // and runs solve() (synchronous on that device).
    const size_t n_dev = gpu_pool_->numDevices();
    const size_t n = emissivity_.size();

    std::vector<std::future<bool>> futures;
    futures.reserve(n_dev);

    // Round-robin assignment: timestep k goes to device k % n_dev. Keeps
    // load balanced when n_steps is not a multiple of n_dev.
    for (size_t d = 0; d < n_dev; ++d) {
        // Build the per-device list of timestep indices to process.
        std::vector<size_t> device_steps;
        for (size_t k = d; k < n_steps; k += n_dev) device_steps.push_back(k);
        if (device_steps.empty()) continue;

        futures.push_back(gpu_pool_->submit(
            [&temperatures_per_timestep, &solar_flux_per_timestep,
             &indirect_per_timestep, device_steps, n,
             &row_norm = vf_row_norm_](
                gpu::opencl::RadiosityOpenCLSolver& solver, size_t /*idx*/) -> bool {
                std::vector<float> temps_f(n), solar_f(n), indirect_f;
                for (size_t k : device_steps) {
                    const auto& temps_d = temperatures_per_timestep[k];
                    const auto& solar_d = solar_flux_per_timestep[k];
                    for (size_t i = 0; i < n; ++i) {
                        temps_f[i] = static_cast<float>(temps_d[i]);
                        solar_f[i] = static_cast<float>(solar_d[i]);
                    }
                    solver.solve(temps_f, solar_f, indirect_f);
                    auto& out = indirect_per_timestep[k];
                    out.resize(n);
                    for (size_t i = 0; i < n; ++i) {
                        const double norm = (i < row_norm.size()) ? row_norm[i] : 1.0;
                        out[i] = static_cast<double>(indirect_f[i]) * norm;
                    }
                }
                return true;
            }));
    }

    bool all_ok = true;
    for (auto& f : futures) {
        if (!f.get()) all_ok = false;
    }
    if (!all_ok) {
        std::cerr << "Warning: one or more multi-GPU radiosity batches "
                     "returned failure; results may be incomplete\n";
    }

    // Aggregate convergence stats from any one solver — they are not strictly
    // tracked across batches, but the caller uses these only for diagnostics.
    last_iterations_ = 0;     // not aggregated; caller treats as advisory
    last_residual_ = 0.0;
    converged_ = all_ok;

    return indirect_per_timestep;
#endif
}

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

    // Compute reflected solar source: ρ_solar_i Q_solar_i (Bond albedo if set, else 1-eps)
    std::vector<double> solar_source(n);
    const bool use_solar_alb = !solar_reflectivity_.empty();
    for (size_t i = 0; i < n; ++i) {
        double rs = use_solar_alb ? solar_reflectivity_[i] : reflectivity_[i];
        solar_source[i] = rs * solar_flux[i];
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

    // incoming_i = Σ_j F_ij B_j  (renormalized so Σ_j F_ij ≤ 1)
    F_op_->matvec(radiosity, incoming);

    for (size_t i = 0; i < n; ++i) {
        // Absorbed flux = emissivity * incoming radiosity
        // (by Kirchhoff's law, absorptivity = emissivity)
        const double norm = (i < vf_row_norm_.size()) ? vf_row_norm_[i] : 1.0;
        indirect_flux[i] = emissivity_[i] * incoming[i] * norm;
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

    // First bounce: ρ_solar_i Q_solar_i leaves face i (Bond albedo if set, else 1-eps)
    std::vector<double> scattered_source(n);
    const bool use_solar_alb = !solar_reflectivity_.empty();
    for (size_t i = 0; i < n; ++i) {
        double rs = use_solar_alb ? solar_reflectivity_[i] : reflectivity_[i];
        scattered_source[i] = rs * solar_flux[i];
    }

    // Received scattered solar: F^T * (ρ Q_solar)  (renormalized so Σ_j F_ij ≤ 1)
    std::vector<double> received(n, 0.0);
    F_op_->matvec(scattered_source, received);

    // Absorbed scattered solar
    std::vector<double> absorbed(n);
    for (size_t i = 0; i < n; ++i) {
        const double norm = (i < vf_row_norm_.size()) ? vf_row_norm_[i] : 1.0;
        absorbed[i] = emissivity_[i] * received[i] * norm;
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
        // Renormalize Σ_j F_ij ≤ 1 so the radiosity gain ρ·Σ stays < 1 (stable).
        for (size_t i = 0; i < n && i < vf_row_norm_.size(); ++i) {
            temp_vec[i] *= vf_row_norm_[i];
        }

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

    // Recenter to the mesh bounding-box center before float32 conversion. At
    // planetary-scale absolute coordinates (e.g. lunar selenocentric, |Z|~1.7e6 m)
    // a float32 ulp (~0.1 m) exceeds a meter-scale patch's features, collapsing
    // Embree's BVH and making visibility ray tracing pathologically slow. The
    // H-matrix builder (HMatrixViewFactor::build) computes the IDENTICAL shift
    // from the same mesh_->vertices and applies it to its ray origins, so rays
    // and scene share one small-coordinate space. View factors depend only on
    // relative geometry, so the constant shift is exact.
    std::array<double,3> bbmin = {{ mesh_->vertices[0][0], mesh_->vertices[0][1], mesh_->vertices[0][2] }};
    std::array<double,3> bbmax = bbmin;
    for (size_t i = 1; i < n_verts; ++i) {
        for (int k = 0; k < 3; ++k) {
            bbmin[k] = std::min(bbmin[k], mesh_->vertices[i][k]);
            bbmax[k] = std::max(bbmax[k], mesh_->vertices[i][k]);
        }
    }
    const std::array<double,3> origin_shift = {{
        0.5 * (bbmin[0] + bbmax[0]),
        0.5 * (bbmin[1] + bbmax[1]),
        0.5 * (bbmin[2] + bbmax[2])
    }};
    std::cout << "  Radiation Embree scene recentered by ("
              << origin_shift[0] << ", " << origin_shift[1] << ", " << origin_shift[2]
              << ") for float32 precision" << std::endl;

    for (size_t i = 0; i < n_verts; ++i) {
        vertices[3 * i + 0] = static_cast<float>(mesh_->vertices[i][0] - origin_shift[0]);
        vertices[3 * i + 1] = static_cast<float>(mesh_->vertices[i][1] - origin_shift[1]);
        vertices[3 * i + 2] = static_cast<float>(mesh_->vertices[i][2] - origin_shift[2]);
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

    // Method selection: Monte Carlo overrides other paths when enabled.
    // MC was introduced because H-matrix build memory is infeasible on 4M+ meshes
    // on this machine (killed by jetsam at ~250 GB RSS) and sparse-VF hits the
    // max_neighbors cap for 99.5% of faces before reaching the energy target.
    bool use_hmatrix = config_.use_hmatrix;
    const bool use_mc = config_.use_monte_carlo_vf;
    std::cout << "  [DEBUG] View factor method selection:" << std::endl;
    std::cout << "    use_monte_carlo_vf = " << (use_mc ? "true" : "false") << std::endl;
    std::cout << "    config_.use_hmatrix = " << (config_.use_hmatrix ? "true" : "false") << std::endl;
    std::cout << "    n_faces = " << n_faces << std::endl;

    if (use_hmatrix && !use_mc && n_faces > 1000000) {
        std::cout << "  Large mesh (>1M faces): H-matrix will use out-of-core storage." << std::endl;
    }

    if (use_mc) {
        std::cout << "Using Monte Carlo ray-gather for view factors..." << std::endl;
        view_factors_ = computeMonteCarlo(mesh_,
                                          static_cast<RTCDevice>(device_),
                                          static_cast<RTCScene>(scene_),
                                          config_,
                                          config_.mc_rays_per_face);
        using_hmatrix_ = false;
        std::cout << "Monte Carlo view factors computed:" << std::endl;
        std::cout << "  Non-zeros: " << view_factors_.numNonZeros() << std::endl;
        std::cout << "  Memory usage: " << view_factors_.memoryUsage() / (1024.0 * 1024.0)
                  << " MB" << std::endl;
    } else if (use_hmatrix) {
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
        hconfig.subdivide_large_admissible = config_.hmatrix_subdivide_admissible;
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
        if (config_.rms_slope_deg > 0.0) {
            chi_w_rad_ = config_.rms_slope_deg * M_PI / 180.0;
            buildChiTable();
            buildBeamPairs();
            beam_active_ = !beam_pairs_.empty();
            configureSolarBeaming();
        }
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

    // Method selection: Monte Carlo overrides when enabled. See
    // computeViewFactorsOnly() for the rationale.
    bool use_hmatrix = config_.use_hmatrix;
    const bool use_mc = config_.use_monte_carlo_vf;
    std::cout << "  [DEBUG] View factor method selection (prepare):" << std::endl;
    std::cout << "    use_monte_carlo_vf = " << (use_mc ? "true" : "false") << std::endl;
    std::cout << "    config_.use_hmatrix = " << (config_.use_hmatrix ? "true" : "false") << std::endl;
    std::cout << "    n_faces = " << n_faces << std::endl;

    if (use_hmatrix && !use_mc && n_faces > 1000000) {
        std::cout << "  Large mesh (>1M faces): H-matrix will use out-of-core storage." << std::endl;
    }

    if (use_mc) {
        std::cout << "Using Monte Carlo ray-gather for view factors..." << std::endl;
        view_factors_ = computeMonteCarlo(mesh_,
                                          static_cast<RTCDevice>(device_),
                                          static_cast<RTCScene>(scene_),
                                          config_,
                                          config_.mc_rays_per_face);
        solver_ = std::make_unique<RadiositySolver>(view_factors_, emissivity_, areas_);
        using_hmatrix_ = false;
        std::cout << "Monte Carlo view factors computed:" << std::endl;
        std::cout << "  Non-zeros: " << view_factors_.numNonZeros() << std::endl;
        std::cout << "  Memory usage: " << view_factors_.memoryUsage() / (1024.0 * 1024.0)
                  << " MB" << std::endl;
    } else if (use_hmatrix) {
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
        hconfig.subdivide_large_admissible = config_.hmatrix_subdivide_admissible;
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

    // Smith (1967) anisotropic beamed-emission setup (only when rms_slope_deg>0).
    if (config_.rms_slope_deg > 0.0) {
        chi_w_rad_ = config_.rms_slope_deg * M_PI / 180.0;
        buildChiTable();
        buildBeamPairs();
        beam_active_ = !beam_pairs_.empty();
        configureSolarBeaming();
    }

    return true;
}

bool RadiationExchangeManager::hasGPUPool() const {
    return solver_ ? solver_->hasGPUPool() : false;
}

std::vector<std::vector<double>>
RadiationExchangeManager::computeIndirectFluxBatch(
    const std::vector<std::vector<double>>& temperatures_per_timestep,
    const std::vector<std::vector<double>>& solar_flux_per_timestep) {
    // Delegate to the Sun-aware overload with no Sun directions (beaming off).
    return computeIndirectFluxBatch(temperatures_per_timestep,
                                    solar_flux_per_timestep, {});
}

std::vector<std::vector<double>>
RadiationExchangeManager::computeIndirectFluxBatch(
    const std::vector<std::vector<double>>& temperatures_per_timestep,
    const std::vector<std::vector<double>>& solar_flux_per_timestep,
    const std::vector<std::array<double, 3>>& sun_dirs) {

    const size_t n_steps = temperatures_per_timestep.size();
    const size_t n = mesh_->faces.size();
    std::vector<std::vector<double>> indirect_per_timestep(n_steps,
                                                            std::vector<double>(n, 0.0));

    if (!is_active_ || !solver_ || n_steps == 0) {
        return indirect_per_timestep;
    }

    auto raw_per_step = solver_->solveBatch(temperatures_per_timestep,
                                             solar_flux_per_timestep);

    const bool do_beam = beam_active_ && (sun_dirs.size() == n_steps);

    // Map back to all faces using the exchange-face index list (same as
    // computeIndirectFlux). Update aggregate stats from the last timestep.
    double sum = 0.0;
    double max_flux = 0.0;
    for (size_t k = 0; k < n_steps; ++k) {
        const auto& flux = raw_per_step[k];
        if (flux.empty()) continue;
        for (size_t i = 0; i < classification_.exchange_face_indices.size(); ++i) {
            int face_idx = classification_.exchange_face_indices[i];
            indirect_per_timestep[k][face_idx] = flux[face_idx];
        }
        // Smith beamed-emission correction (per timestep, Sun-dependent).
        if (do_beam) {
            const auto& s = sun_dirs[k];
            if (s[0] != 0.0 || s[1] != 0.0 || s[2] != 0.0) {
                addBeamedFlux(temperatures_per_timestep[k], solar_flux_per_timestep[k],
                              s, indirect_per_timestep[k]);
            }
        }
    }
    if (!raw_per_step.empty() && !raw_per_step.back().empty()) {
        for (int idx : classification_.exchange_face_indices) {
            sum += indirect_per_timestep.back()[idx];
            max_flux = std::max(max_flux, indirect_per_timestep.back()[idx]);
        }
        stats_.mean_indirect_flux = sum / std::max(size_t(1),
                                                    classification_.exchange_faces);
        stats_.max_indirect_flux = max_flux;
        stats_.solver_iterations = solver_->getLastIterations();
        stats_.solver_residual = solver_->getLastResidual();
    }

    return indirect_per_timestep;
}

std::vector<double> RadiationExchangeManager::computeIndirectFlux(
    const std::vector<double>& temperatures,
    const std::vector<double>& solar_flux) {
    // Delegate to the Sun-aware overload with no Sun direction (beaming off).
    return computeIndirectFlux(temperatures, solar_flux, {0.0, 0.0, 0.0});
}

std::vector<double> RadiationExchangeManager::computeIndirectFlux(
    const std::vector<double>& temperatures,
    const std::vector<double>& solar_flux,
    const std::array<double, 3>& sun_dir) {

    size_t n = mesh_->faces.size();
    indirect_flux_.assign(n, 0.0);

    if (!is_active_ || !solver_) {
        return indirect_flux_;
    }

    // Solve radiosity equation (isotropic/Lambertian thermal emission)
    auto flux = solver_->solve(temperatures, solar_flux);

    // Map back to all faces
    for (size_t i = 0; i < classification_.exchange_face_indices.size(); ++i) {
        int face_idx = classification_.exchange_face_indices[i];
        indirect_flux_[face_idx] = flux[face_idx];
    }

    // Smith (1967) anisotropic beamed-emission correction (Sun-dependent).
    // Redistributes each sunlit wall's fixed total emission toward the floor.
    const bool sun_valid = (sun_dir[0] != 0.0 || sun_dir[1] != 0.0 || sun_dir[2] != 0.0);
    if (beam_active_ && sun_valid) {
        addBeamedFlux(temperatures, solar_flux, sun_dir, indirect_flux_);
    }

    // Update statistics (after beaming so they reflect the corrected flux)
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

// ============================================================================
// Smith (1967) anisotropic beamed thermal emission
// The directional-emission math lives in smith_beaming.h (shared with the
// standalone validator test_beam_chi.cpp).
// ============================================================================

void RadiationExchangeManager::buildChiTable() {
    chi_ns_ = 24; chi_ne_ = 24; chi_npsi_ = 33;
    chi_table_ = smith::buildChiTable(chi_w_rad_, chi_ns_, chi_ne_, chi_npsi_);
}

float RadiationExchangeManager::lookupChi(double cos_sun, double cos_e,
                                          double cos_psi) const {
    return smith::evalChi(chi_table_, chi_ns_, chi_ne_, chi_npsi_,
                          cos_sun, cos_e, cos_psi);
}

void RadiationExchangeManager::buildBeamPairs() {
    beam_pairs_.clear();
    if (classification_.exchange_face_indices.empty()) return;

    // Receivers = the radiation-exchange (shadowed-floor) faces.
    const std::vector<int>& receivers = classification_.exchange_face_indices;

    // Geometric VF + Embree visibility for receiver rows. Backend-independent
    // (works for H-matrix runs where per-row matrix access is unavailable), and
    // uses the same F formula and row-normalization as the radiosity.
    ViewFactorCalculator calc(mesh_, static_cast<RTCDevice>(device_),
                              static_cast<RTCScene>(scene_), config_);
    ViewFactorMatrix rows = calc.computeSubset(receivers);  // global row indexing

    // Cache per-face mean normals (float) for the run-time beamed pass.
    const auto& normals = calc.getFaceNormals();
    const auto& centers = calc.getFaceCenters();
    beam_normals_.resize(normals.size());
    for (size_t f = 0; f < normals.size(); ++f) {
        beam_normals_[f] = { static_cast<float>(normals[f][0]),
                             static_cast<float>(normals[f][1]),
                             static_cast<float>(normals[f][2]) };
    }

    const auto& shadow = classification_.shadow_fractions;
    const double thresh = config_.beam_vf_threshold;

    std::vector<int> cols;
    std::vector<double> vals;
    for (int i : receivers) {
        rows.getRow(i, cols, vals);     // global receiver index
        if (cols.empty()) continue;
        double rowsum = 0.0;
        for (double v : vals) rowsum += v;
        double rownorm = 1.0 / std::max(1.0, rowsum);   // matches radiosity vf_row_norm
        const auto& ci = centers[i];
        for (size_t m = 0; m < cols.size(); ++m) {
            int j = cols[m];
            if (j < 0 || j >= static_cast<int>(shadow.size())) continue;
            if (shadow[j] >= 0.99) continue;            // permanently shadowed -> not an emitter
            double Fhat = vals[m] * rownorm;
            if (Fhat <= thresh) continue;
            const auto& cj = centers[j];
            double dx = ci[0] - cj[0], dy = ci[1] - cj[1], dz = ci[2] - cj[2];
            double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 < 1e-20) continue;
            double inv = 1.0 / std::sqrt(r2);
            double dirx = dx * inv, diry = dy * inv, dirz = dz * inv;
            const auto& nj = normals[j];
            double cos_e = nj[0] * dirx + nj[1] * diry + nj[2] * dirz;
            if (cos_e <= 0.0) continue;                 // wall not emitting toward floor
            BeamPair bp;
            bp.i = i;
            bp.j = j;
            bp.Fhat_ij = static_cast<float>(Fhat);
            bp.cos_e = static_cast<float>(cos_e);
            bp.dir_ji = { static_cast<float>(dirx),
                          static_cast<float>(diry),
                          static_cast<float>(dirz) };
            beam_pairs_.push_back(bp);
        }
    }

    std::cout << "  Smith beamed emission: " << beam_pairs_.size()
              << " wall->floor beam pairs (rms_slope = " << config_.rms_slope_deg
              << " deg)" << std::endl;
}

void RadiationExchangeManager::configureSolarBeaming() {
    beam_solar_ = false;
    solar_albedo_beam_ = 0.0;
    if (const char* bs = std::getenv("PTM_BEAM_SOLAR")) beam_solar_ = (std::atoi(bs) != 0);
    if (const char* sa = std::getenv("PTM_SOLAR_ALBEDO")) {
        double a = std::atof(sa);
        if (a > 0.0 && a < 1.0) solar_albedo_beam_ = a;
    }
    if (beam_solar_ && solar_albedo_beam_ <= 0.0) solar_albedo_beam_ = 0.12;  // default Bond
    if (beam_solar_) {
        std::cout << "  [PTM_BEAM_SOLAR] beaming reflected solar (Bond albedo "
                  << solar_albedo_beam_ << ") toward the floor, in addition to thermal IR"
                  << std::endl;
    }
}

void RadiationExchangeManager::addBeamedFlux(const std::vector<double>& temperatures,
                                             const std::vector<double>& solar_flux,
                                             const std::array<double, 3>& sun_dir,
                                             std::vector<double>& flux) const {
    if (beam_pairs_.empty()) return;
    const float sx = static_cast<float>(sun_dir[0]);
    const float sy = static_cast<float>(sun_dir[1]);
    const float sz = static_cast<float>(sun_dir[2]);

    // Accumulate the signed (chi-1) corrections, then apply once per receiver so
    // the result stays non-negative (absorbed flux). OpenMP-safe via atomic add
    // on the receiver bucket would race the final clamp, so reduce into a local.
    const size_t np = beam_pairs_.size();
    #pragma omp parallel for schedule(static)
    for (size_t k = 0; k < np; ++k) {
        const BeamPair& bp = beam_pairs_[k];
        const std::array<float, 3>& nj = beam_normals_[bp.j];
        double cos_sun = static_cast<double>(nj[0] * sx + nj[1] * sy + nj[2] * sz);
        if (cos_sun <= 0.0) continue;   // wall mean in shadow this timestep -> no beaming
        double cos_psi = static_cast<double>(bp.dir_ji[0] * sx + bp.dir_ji[1] * sy
                                             + bp.dir_ji[2] * sz);
        double chi = lookupChi(cos_sun, static_cast<double>(bp.cos_e), cos_psi);
        double delta = chi - 1.0;
        if (delta == 0.0) continue;
        double Tj = temperatures[bp.j];
        double Ej = emissivity_[bp.j] * SIGMA_SB * Tj * Tj * Tj * Tj;   // thermal IR source
        // Solar opposition beaming (PTM_BEAM_SOLAR): the reflected-solar source
        // (albedo * incident sunlight) leaving the sunlit wall has the same m.s facet
        // structure as its IR emission, so the floor (backscatter) gets the same chi
        // enhancement. Absorbed at the floor with eps_i (consistent with the radiosity's
        // solar handling). Off by default => byte-identical IR-only beaming.
        double Sj = 0.0;
        if (beam_solar_ && bp.j < static_cast<int>(solar_flux.size())) {
            Sj = solar_albedo_beam_ * solar_flux[bp.j];
        }
        double dq = emissivity_[bp.i] * (Ej + Sj) * static_cast<double>(bp.Fhat_ij) * delta;
        #pragma omp atomic
        flux[bp.i] += dq;
    }

    // Keep absorbed indirect flux non-negative on the receiver set.
    for (int i : classification_.exchange_face_indices) {
        if (flux[i] < 0.0) flux[i] = 0.0;
    }
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
        // For large meshes, use memory-mapped loading to avoid loading entire matrix into RAM.
        //
        // KNOWN BUG (2026-04): parallel matvec on a mmap-loaded H-matrix SIGSEGVs
        // during the first radiosity iteration on >1M-face meshes. Root cause
        // under investigation. Until it's fixed, auto-mmap is effectively
        // disabled by raising the threshold far above any expected mesh size.
        // Full-RAM load is safe and the matrix typically fits comfortably
        // (probe: 1.57M faces → 2.1 GB .hmat ≪ 384 GB RAM).
        size_t n_faces = mesh_->faces.size();
        const size_t MMAP_THRESHOLD_FACES = 1'000'000'000;  // effectively off

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
