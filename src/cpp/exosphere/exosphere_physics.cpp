/*
 * Europa Exosphere Physics Module - macOS Compatible Implementation
 * Fixes AVX2/AVX512 and Eigen issues for AppleClang
 */

#include "exosphere_physics.h"
#include <algorithm>
#include <numeric>
#include <chrono>
#include <omp.h>
#include <immintrin.h>  // For AVX2 intrinsics

// Check for AVX support at compile time
#ifdef __AVX512F__
    #define USE_AVX512
#elif defined(__AVX2__)
    #define USE_AVX2
#endif

namespace ptm {
namespace exosphere {

// VaporPressure vector operations - using scalar implementations on macOS
void VaporPressure::calculateVaporPressureVector(const std::vector<double>& temperatures,
                                                std::vector<double>& pressures) {
    pressures.resize(temperatures.size());
    
    #pragma omp parallel for
    for (size_t i = 0; i < temperatures.size(); ++i) {
        pressures[i] = calculateVaporPressure(temperatures[i]);
    }
}

void VaporPressure::calculateSublimationFluxVector(const std::vector<double>& temperatures,
                                                  std::vector<double>& fluxes) {
    fluxes.resize(temperatures.size());
    
    #pragma omp parallel for
    for (size_t i = 0; i < temperatures.size(); ++i) {
        fluxes[i] = calculateSublimationFlux(temperatures[i]);
    }
}

#ifdef AVX2_ENABLED
// Simplified AVX2 for thermal velocity (basic arithmetic only)
void VaporPressure::calculateThermalVelocityVectorAVX2(const double* temperatures,
                                                       double* velocities,
                                                       size_t count) {
    const __m256d factor = _mm256_set1_pd(8.0 * constants::BOLTZMANN_K / 
                                          (M_PI * constants::WATER_MOLECULAR_MASS));
    
    size_t i = 0;
    // Process 4 elements at a time with AVX2
    for (; i + 3 < count; i += 4) {
        __m256d T = _mm256_loadu_pd(&temperatures[i]);
        __m256d v_th = _mm256_sqrt_pd(_mm256_mul_pd(factor, T));
        _mm256_storeu_pd(&velocities[i], v_th);
    }
    
    // Handle remaining elements
    for (; i < count; ++i) {
        velocities[i] = calculateThermalVelocity(temperatures[i]);
    }
}
#endif

// BallisticTransport implementations
BallisticTransport::BallisticTransport(double gravity, double radius, double escape_v)
    : gravity_(gravity), body_radius_(radius), escape_velocity_(escape_v),
      cache_valid_(false),
      optimized_cache_(radius),
      optimized_cache_built_(false),
      rebuild_controller_(0.05, 2.0),
      spatial_index_built_(false),
      sinkhorn_cache_valid_(false) {}

double BallisticTransport::calculateEscapeFraction(double temperature_K) const {
    if (temperature_K <= 0.0) return 0.0;
    
    // Jeans escape parameter
    double lambda = (escape_velocity_ * escape_velocity_ * constants::WATER_MOLECULAR_MASS) / 
                   (2.0 * constants::BOLTZMANN_K * temperature_K);
    
    // Jeans escape flux approximation
    if (lambda > 50.0) {
        return 0.0;  // Negligible escape
    } else if (lambda < 2.0) {
        return 0.5;  // Significant escape
    } else {
        // Chamberlain (1963) approximation
        double escape_rate = std::exp(-lambda) * (1.0 + lambda);
        double thermal_rate = 2.0 * std::sqrt(M_PI * lambda);
        return escape_rate / thermal_rate;
    }
}

double BallisticTransport::transportProbabilityIntegrated(double distance_m, 
                                                         double solid_angle,
                                                         double temperature_K) const {
    double p_density = transportProbabilityDensity(distance_m, temperature_K);
    return p_density * solid_angle;
}

double BallisticTransport::calculateSolidAngle(double face_area, double distance) const {
    if (distance <= 0.0) return 0.0;
    return face_area / (distance * distance);
}

void BallisticTransport::verifyMassConservation(const Eigen::SparseMatrix<double>& matrix) const {
    size_t n_faces = matrix.cols();
    
    double max_column_sum = 0.0;
    double min_column_sum = 1.0;
    double avg_column_sum = 0.0;
    
    for (size_t j = 0; j < n_faces; ++j) {
        double col_sum = 0.0;
        for (Eigen::SparseMatrix<double>::InnerIterator it(matrix, j); it; ++it) {
            col_sum += it.value();
        }
        max_column_sum = std::max(max_column_sum, col_sum);
        min_column_sum = std::min(min_column_sum, col_sum);
        avg_column_sum += col_sum;
    }
    avg_column_sum /= n_faces;
    
    std::cout << "Mass conservation check:" << std::endl;
    std::cout << "  Column sums: min=" << std::setprecision(6) << min_column_sum 
              << ", max=" << max_column_sum 
              << ", avg=" << avg_column_sum << std::endl;
    
    if (max_column_sum > 1.0 + 1e-6) {
        std::cerr << "WARNING: Mass conservation violated! Max column sum = " 
                  << max_column_sum << std::endl;
    }
}

void BallisticTransport::buildTransportMatrixOptimized(
    const std::vector<std::array<double, 3>>& face_centers,
    const std::vector<double>& face_areas,
    const std::vector<double>& temperatures,
    Eigen::SparseMatrix<double>& transport_matrix,
    bool use_cache) {
    
    size_t n_faces = face_centers.size();
    
    // Pre-compute all pairwise distances if not cached
    if (!cache_valid_ || !use_cache) {
        distance_cache_.resize(n_faces);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < n_faces; ++i) {
            distance_cache_[i].resize(n_faces);
            for (size_t j = 0; j < n_faces; ++j) {
                if (i != j) {
                    distance_cache_[i][j] = greatCircleDistance(face_centers[i], face_centers[j]);
                } else {
                    distance_cache_[i][j] = 0.0;
                }
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Distance cache computation took " << duration.count() << " ms" << std::endl;
        
        cache_valid_ = true;
    }
    
    // Build transport matrix
    transport_matrix = buildTransportMatrix(face_centers, face_areas, temperatures);
}

// Constructor with initial ice distribution
ExosphereModel::ExosphereModel(size_t num_faces) 
    : num_faces_(num_faces),
      transport_matrix_valid_(false),
      time_step_(3600.0),
      convergence_threshold_(1e-6),
      max_iterations_(1000),
      total_mass_sublimated_(0.0),
      total_mass_deposited_(0.0),
      total_mass_escaped_(0.0),
      initial_equatorial_thickness_(1e-4),  // ADD THIS - default 0.1mm
      initial_polar_thickness_(1e-3) {      // ADD THIS - default 1mm
    
    state_ = std::make_shared<ExosphereState>(num_faces);
    transport_ = std::make_unique<BallisticTransport>();
    
    // Initialize with uniform ice distribution (1 mm thick layer everywhere)
    const double initial_ice_thickness = 1e-3;  // 1 mm in meters
    const double initial_mass = initial_ice_thickness * constants::ICE_DENSITY;
    
    #ifdef USE_AVX512
        const __m512d ice_mass = _mm512_set1_pd(initial_mass);
        const __m512d ice_thick = _mm512_set1_pd(initial_ice_thickness);
        size_t i = 0;
        for (; i + 8 <= num_faces; i += 8) {
            _mm512_storeu_pd(&state_->accumulated_mass[i], ice_mass);
            _mm512_storeu_pd(&state_->ice_thickness[i], ice_thick);
        }
        // Handle remainder
        for (; i < num_faces; ++i) {
            state_->accumulated_mass[i] = initial_mass;
            state_->ice_thickness[i] = initial_ice_thickness;
        }
    #elif defined(USE_AVX2)
        const __m256d ice_mass = _mm256_set1_pd(initial_mass);
        const __m256d ice_thick = _mm256_set1_pd(initial_ice_thickness);
        size_t i = 0;
        for (; i + 4 <= num_faces; i += 4) {
            _mm256_storeu_pd(&state_->accumulated_mass[i], ice_mass);
            _mm256_storeu_pd(&state_->ice_thickness[i], ice_thick);
        }
        // Handle remainder
        for (; i < num_faces; ++i) {
            state_->accumulated_mass[i] = initial_mass;
            state_->ice_thickness[i] = initial_ice_thickness;
        }
    #else
        for (size_t i = 0; i < num_faces; ++i) {
            state_->accumulated_mass[i] = initial_mass;
            state_->ice_thickness[i] = initial_ice_thickness;
        }
    #endif
}

// Vectorized helper function for vapor pressure calculation with ice check
#ifdef USE_AVX512
void updateEmissionWithIceCheckAVX512(
    const double* temperatures,
    const double* accumulated_mass,
    double* vapor_pressure,
    double* thermal_velocity,
    double* vapor_density,
    double* emission_flux,
    size_t n) {
    
    const __m512d zero = _mm512_setzero_pd();
    const __m512d quarter = _mm512_set1_pd(0.25);
    const __m512d k_B = _mm512_set1_pd(constants::BOLTZMANN_K);
    const __m512d m_H2O = _mm512_set1_pd(constants::WATER_MOLECULAR_MASS);
    const __m512d pi_val = _mm512_set1_pd(M_PI);
    
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        // Load data
        __m512d T = _mm512_loadu_pd(&temperatures[i]);
        __m512d mass = _mm512_loadu_pd(&accumulated_mass[i]);
        
        // Calculate vapor pressure (simplified - you'd call actual function)
        __m512d P_vap = _mm512_loadu_pd(&vapor_pressure[i]); // Assume pre-calculated
        
        // Calculate thermal velocity: v_th = sqrt(8*k_B*T/(pi*m))
        __m512d v_th_sq = _mm512_mul_pd(_mm512_set1_pd(8.0), k_B);
        v_th_sq = _mm512_mul_pd(v_th_sq, T);
        v_th_sq = _mm512_div_pd(v_th_sq, _mm512_mul_pd(pi_val, m_H2O));
        __m512d v_th = _mm512_sqrt_pd(v_th_sq);
        
        // Calculate vapor MASS density: rho = P*M/(R*T) = P/(k_B*T)*m_molecule
        // Note: P/(k_B*T) gives number density, multiply by m_H2O to get mass density
        __m512d rho_v = _mm512_mul_pd(_mm512_div_pd(P_vap, _mm512_mul_pd(k_B, T)), m_H2O);
        
        // Calculate theoretical emission: 0.25 * rho * v_th
        __m512d emission_theoretical = _mm512_mul_pd(quarter, _mm512_mul_pd(rho_v, v_th));
        
        // CRITICAL: Apply ice availability mask
        __mmask8 ice_available = _mm512_cmp_pd_mask(mass, zero, _CMP_GT_OQ);
        __m512d emission_actual = _mm512_mask_blend_pd(ice_available, zero, emission_theoretical);
        
        // Store results
        _mm512_storeu_pd(&thermal_velocity[i], v_th);
        _mm512_storeu_pd(&vapor_density[i], rho_v);
        _mm512_storeu_pd(&emission_flux[i], emission_actual);
    }
    
    // Handle remainder with scalar code
    for (; i < n; ++i) {
        double T = temperatures[i];
        thermal_velocity[i] = std::sqrt(8.0 * constants::BOLTZMANN_K * T /
                                       (M_PI * constants::WATER_MOLECULAR_MASS));
        // FIXED: vapor_density is MASS density (kg/m³), not number density
        // Using ideal gas law: ρ = P*M/(R*T) which equals P/(k_B*T)*m_molecule
        vapor_density[i] = vapor_pressure[i] * constants::WATER_MOLAR_MASS /
                          (constants::GAS_CONSTANT_R * T);

        double theoretical_emission = 0.25 * vapor_density[i] * thermal_velocity[i];
        emission_flux[i] = (accumulated_mass[i] > 0.0) ? theoretical_emission : 0.0;
    }
}
#endif

#ifdef USE_AVX2
void updateEmissionWithIceCheckAVX2(
    const double* temperatures,
    const double* accumulated_mass,
    double* vapor_pressure,
    double* thermal_velocity,
    double* vapor_density,
    double* emission_flux,
    size_t n) {
    
    const __m256d zero = _mm256_setzero_pd();
    const __m256d quarter = _mm256_set1_pd(0.25);
    const __m256d k_B = _mm256_set1_pd(constants::BOLTZMANN_K);
    const __m256d m_H2O = _mm256_set1_pd(constants::WATER_MOLECULAR_MASS);
    const __m256d pi_val = _mm256_set1_pd(M_PI);
    
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        // Load data
        __m256d T = _mm256_loadu_pd(&temperatures[i]);
        __m256d mass = _mm256_loadu_pd(&accumulated_mass[i]);
        
        // Calculate vapor pressure (simplified - assume pre-calculated)
        __m256d P_vap = _mm256_loadu_pd(&vapor_pressure[i]);
        
        // Calculate thermal velocity: v_th = sqrt(8*k_B*T/(pi*m))
        __m256d v_th_sq = _mm256_mul_pd(_mm256_set1_pd(8.0), k_B);
        v_th_sq = _mm256_mul_pd(v_th_sq, T);
        v_th_sq = _mm256_div_pd(v_th_sq, _mm256_mul_pd(pi_val, m_H2O));
        __m256d v_th = _mm256_sqrt_pd(v_th_sq);
        
        // Calculate vapor MASS density: rho = P*M/(R*T) = P/(k_B*T)*m_molecule
        // Note: P/(k_B*T) gives number density, multiply by m_H2O to get mass density
        __m256d rho_v = _mm256_mul_pd(_mm256_div_pd(P_vap, _mm256_mul_pd(k_B, T)), m_H2O);
        
        // Calculate theoretical emission: 0.25 * rho * v_th
        __m256d emission_theoretical = _mm256_mul_pd(quarter, _mm256_mul_pd(rho_v, v_th));
        
        // CRITICAL: Apply ice availability mask
        __m256d ice_mask = _mm256_cmp_pd(mass, zero, _CMP_GT_OQ);
        __m256d emission_actual = _mm256_and_pd(emission_theoretical, ice_mask);
        
        // Store results
        _mm256_storeu_pd(&thermal_velocity[i], v_th);
        _mm256_storeu_pd(&vapor_density[i], rho_v);
        _mm256_storeu_pd(&emission_flux[i], emission_actual);
    }
    
    // Handle remainder with scalar code
    for (; i < n; ++i) {
        double T = temperatures[i];
        thermal_velocity[i] = std::sqrt(8.0 * constants::BOLTZMANN_K * T /
                                       (M_PI * constants::WATER_MOLECULAR_MASS));
        // FIXED: vapor_density is MASS density (kg/m³), not number density
        // Using ideal gas law: ρ = P*M/(R*T) which equals P/(k_B*T)*m_molecule
        vapor_density[i] = vapor_pressure[i] * constants::WATER_MOLAR_MASS /
                          (constants::GAS_CONSTANT_R * T);

        double theoretical_emission = 0.25 * vapor_density[i] * thermal_velocity[i];
        emission_flux[i] = (accumulated_mass[i] > 0.0) ? theoretical_emission : 0.0;
    }
}
#endif

void ExosphereModel::setMeshGeometry(const std::vector<std::array<double, 3>>& face_centers,
                                    const std::vector<double>& face_areas) {
    face_centers_ = face_centers;
    face_areas_ = face_areas;
    transport_matrix_valid_ = false;
    
    calculateFaceCoordinates();
}

void ExosphereModel::calculateFaceCoordinates() {
    face_latitudes_.resize(num_faces_);
    face_longitudes_.resize(num_faces_);
    
    #pragma omp parallel for
    for (size_t i = 0; i < num_faces_; ++i) {
        double x = face_centers_[i][0];
        double y = face_centers_[i][1];
        double z = face_centers_[i][2];
        double r = std::sqrt(x*x + y*y + z*z);
        
        face_latitudes_[i] = std::asin(z / r) * 180.0 / M_PI;
        face_longitudes_[i] = std::atan2(y, x) * 180.0 / M_PI;
    }
}

void ExosphereModel::updateTemperatures(const std::vector<double>& temperatures) {
    if (temperatures.size() != num_faces_) {
        throw std::runtime_error("Temperature array size mismatch");
    }
    
    // First calculate vapor pressures (can use existing vectorized function)
    #pragma omp parallel for
    for (size_t i = 0; i < num_faces_; ++i) {
        state_->vapor_pressure[i] = VaporPressure::calculateVaporPressure(temperatures[i]);
    }
    
    // Use vectorized emission calculation with ice check
    #ifdef USE_AVX512
        updateEmissionWithIceCheckAVX512(
            temperatures.data(),
            state_->accumulated_mass.data(),
            state_->vapor_pressure.data(),
            state_->thermal_velocity.data(),
            state_->vapor_density.data(),
            state_->emission_flux.data(),
            num_faces_
        );
    #elif defined(USE_AVX2)
        updateEmissionWithIceCheckAVX2(
            temperatures.data(),
            state_->accumulated_mass.data(),
            state_->vapor_pressure.data(),
            state_->thermal_velocity.data(),
            state_->vapor_density.data(),
            state_->emission_flux.data(),
            num_faces_
        );
    #else
        // Scalar fallback with OpenMP
        #pragma omp parallel for
        for (size_t i = 0; i < num_faces_; ++i) {
            state_->thermal_velocity[i] = VaporPressure::calculateThermalVelocity(temperatures[i]);
            state_->vapor_density[i] = VaporPressure::calculateVaporDensity(
                state_->vapor_pressure[i], temperatures[i]);
            
            // Theoretical emission flux
            double theoretical_emission = state_->vapor_density[i] * state_->thermal_velocity[i] / 4.0;
            
            // CRITICAL FIX: Check for available ice
            if (state_->accumulated_mass[i] > 0.0) {
                state_->emission_flux[i] = theoretical_emission;
            } else {
                state_->emission_flux[i] = 0.0;
            }
        }
    #endif
    
    // Calculate escape fluxes
    #pragma omp parallel for
    for (size_t i = 0; i < num_faces_; ++i) {
        double escape_frac = transport_->calculateEscapeFraction(temperatures[i]);
        state_->escape_flux[i] = state_->emission_flux[i] * escape_frac;
    }

    // Store temperatures for matrix-free transport in evolveTimeStep()
    last_temperatures_ = temperatures;

    // Matrix-free transport is used in evolveTimeStep() — skip O(N²) matrix build.
    // The full sparse matrix can exceed 100 GB for meshes with >100K faces.
    // Matrix build can still be forced via EXO_USE_MATRIX=1 for small meshes.
    {
        bool use_matrix = false;
        if (const char* envp = std::getenv("EXO_USE_MATRIX")) {
            use_matrix = (std::atoi(envp) != 0);
        }
        if (!use_matrix || skip_matrix_rebuild_) {
            return;
        }
    }

    // Check environment variable for which matrix build method to use
    bool use_cached = true;
    if (const char* envp = std::getenv("EXO_NO_CACHE")) {
        use_cached = (std::atoi(envp) == 0);
    }

    // Check if smooth kernel is requested (no shell binning)
    // Default: ON - smooth kernel avoids shell-binning artifacts
    bool use_smooth_kernel = true;
    if (const char* envp = std::getenv("EXO_USE_SMOOTH_KERNEL")) {
        use_smooth_kernel = (std::atoi(envp) != 0);
    }

    // Check if adaptive rebuild is enabled (default: yes)
    bool use_adaptive = true;
    if (const char* envp = std::getenv("EXO_NO_ADAPTIVE")) {
        use_adaptive = (std::atoi(envp) == 0);
    }

    // Only rebuild transport matrix if needed (adaptive) or forced
    bool needs_rebuild = true;
    if (use_adaptive) {
        needs_rebuild = transport_->needsMatrixRebuild(temperatures);
        if (!needs_rebuild) {
            // Skip rebuild, matrix is still valid from previous timestep
            return;
        }
    }

    // Rebuild transport matrix using selected method
    if (use_smooth_kernel) {
        if (verbose_) {
            std::cout << "Building transport matrix (SMOOTH kernel)..." << std::flush;
        }
        transport_matrix_ = transport_->buildTransportMatrixSmooth(face_centers_, face_areas_, temperatures);
        if (verbose_) {
            std::cout << " done (" << num_faces_ << " faces, "
                      << transport_matrix_.nonZeros() << " non-zeros)" << std::endl;
        }
    } else if (use_cached) {
        transport_matrix_ = transport_->buildTransportMatrixCached(face_centers_, face_areas_, temperatures);
    } else {
        transport_matrix_ = transport_->buildTransportMatrix(face_centers_, face_areas_, temperatures);
    }
    transport_matrix_valid_ = true;
}

// Vectorized mass update for time evolution
#ifdef USE_AVX512
void updateMassAVX512(
    double* accumulated_mass,
    double* ice_thickness,
    const double* net_flux,
    double dt,
    size_t n) {
    
    const __m512d zero = _mm512_setzero_pd();
    const __m512d dt_vec = _mm512_set1_pd(dt);
    const __m512d ice_density_inv = _mm512_set1_pd(1.0 / constants::ICE_DENSITY);
    
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        // Load current mass and net flux
        __m512d mass = _mm512_loadu_pd(&accumulated_mass[i]);
        __m512d flux = _mm512_loadu_pd(&net_flux[i]);
        
        // Calculate mass change: dm = net_flux * dt
        __m512d dm = _mm512_mul_pd(flux, dt_vec);
        
        // Update mass: new_mass = old_mass + dm
        __m512d new_mass = _mm512_add_pd(mass, dm);
        
        // Ensure non-negative (max with zero)
        new_mass = _mm512_max_pd(new_mass, zero);
        
        // Calculate ice thickness
        __m512d thickness = _mm512_mul_pd(new_mass, ice_density_inv);
        
        // Store results
        _mm512_storeu_pd(&accumulated_mass[i], new_mass);
        _mm512_storeu_pd(&ice_thickness[i], thickness);
    }
    
    // Handle remainder
    for (; i < n; ++i) {
        double new_mass = accumulated_mass[i] + net_flux[i] * dt;
        accumulated_mass[i] = std::max(0.0, new_mass);
        ice_thickness[i] = accumulated_mass[i] / constants::ICE_DENSITY;
    }
}
#endif

#ifdef USE_AVX2
void updateMassAVX2(
    double* accumulated_mass,
    double* ice_thickness,
    const double* net_flux,
    double dt,
    size_t n) {
    
    const __m256d zero = _mm256_setzero_pd();
    const __m256d dt_vec = _mm256_set1_pd(dt);
    const __m256d ice_density_inv = _mm256_set1_pd(1.0 / constants::ICE_DENSITY);
    
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        // Load current mass and net flux
        __m256d mass = _mm256_loadu_pd(&accumulated_mass[i]);
        __m256d flux = _mm256_loadu_pd(&net_flux[i]);
        
        // Calculate mass change: dm = net_flux * dt
        __m256d dm = _mm256_mul_pd(flux, dt_vec);
        
        // Update mass: new_mass = old_mass + dm
        __m256d new_mass = _mm256_add_pd(mass, dm);
        
        // Ensure non-negative (max with zero)
        new_mass = _mm256_max_pd(new_mass, zero);
        
        // Calculate ice thickness
        __m256d thickness = _mm256_mul_pd(new_mass, ice_density_inv);
        
        // Store results
        _mm256_storeu_pd(&accumulated_mass[i], new_mass);
        _mm256_storeu_pd(&ice_thickness[i], thickness);
    }
    
    // Handle remainder
    for (; i < n; ++i) {
        double new_mass = accumulated_mass[i] + net_flux[i] * dt;
        accumulated_mass[i] = std::max(0.0, new_mass);
        ice_thickness[i] = accumulated_mass[i] / constants::ICE_DENSITY;
    }
}
#endif

void ExosphereModel::updateTransportMatrix(bool force_rebuild) {
    if (!transport_matrix_valid_ || force_rebuild) {
        std::vector<double> temperatures(num_faces_);
        
        #pragma omp parallel for
        for (size_t i = 0; i < num_faces_; ++i) {
            double v_th = state_->thermal_velocity[i];
            if (v_th > 0.0) {
                temperatures[i] = (v_th * v_th * M_PI * constants::WATER_MOLECULAR_MASS) / 
                                (8.0 * constants::BOLTZMANN_K);
            } else {
                temperatures[i] = 0.0;
            }
        }
        
        transport_->buildTransportMatrixOptimized(face_centers_, face_areas_, 
                                                 temperatures, transport_matrix_, true);
        transport_matrix_valid_ = true;
    }
}

// Vectorized latitude-dependent ice distribution
void ExosphereModel::setLatitudinalIceDistribution(double equatorial_thickness, 
                                                   double polar_thickness) {
    // Validate input parameters
    if (equatorial_thickness < 0.0 || polar_thickness < 0.0) {
        throw std::invalid_argument("Ice thickness values must be non-negative");
    }
    
    if (equatorial_thickness == 0.0 && polar_thickness == 0.0) {
        throw std::invalid_argument("At least one thickness value must be positive");
    }
    
    // Ensure face coordinates (latitudes/longitudes) are calculated
    if (face_latitudes_.empty() || face_longitudes_.empty()) {
        calculateFaceCoordinates();
    }
    
    // Calculate thickness difference for interpolation
    const double thickness_diff = polar_thickness - equatorial_thickness;
    
    // Statistics tracking
    double min_thickness = std::numeric_limits<double>::max();
    double max_thickness = 0.0;
    double total_mass_before = getTotalIceMass();  // Store previous mass if any
    
    // Choose implementation based on available vectorization
#ifdef __AVX512F__
    // AVX512 implementation with transcendental functions
    {
        const __m512d eq_thick = _mm512_set1_pd(equatorial_thickness);
        const __m512d thick_diff = _mm512_set1_pd(thickness_diff);
        const __m512d ice_dens = _mm512_set1_pd(constants::ICE_DENSITY);
        const __m512d pi_180 = _mm512_set1_pd(M_PI / 180.0);
        const __m512d two = _mm512_set1_pd(2.0);
        
        size_t i = 0;
        for (; i + 8 <= num_faces_; i += 8) {
            // Load latitudes in degrees
            __m512d lat_deg = _mm512_loadu_pd(&face_latitudes_[i]);
            
            // Convert to radians
            __m512d lat_rad = _mm512_mul_pd(lat_deg, pi_180);
            
            // Calculate sin(latitude)
            // Note: _mm512_sin_pd requires SVML or custom implementation
            // For portability, we'll compute in scalar and load
            alignas(64) double sin_vals[8];
            for (int j = 0; j < 8; ++j) {
                sin_vals[j] = std::sin(face_latitudes_[i + j] * M_PI / 180.0);
            }
            __m512d sin_lat = _mm512_load_pd(sin_vals);
            
            // Calculate abs(sin(lat))^2
            __m512d abs_sin_lat = _mm512_abs_pd(sin_lat);
            __m512d sin_lat_sq = _mm512_mul_pd(abs_sin_lat, abs_sin_lat);
            
            // Interpolate thickness: equatorial + diff * sin^2(lat)
            __m512d thickness = _mm512_fmadd_pd(thick_diff, sin_lat_sq, eq_thick);
            
            // Calculate mass: thickness * ice_density
            __m512d mass = _mm512_mul_pd(thickness, ice_dens);
            
            // Store results
            _mm512_storeu_pd(&state_->ice_thickness[i], thickness);
            _mm512_storeu_pd(&state_->accumulated_mass[i], mass);
            
            // Update statistics (scalar for simplicity)
            for (int j = 0; j < 8 && (i + j) < num_faces_; ++j) {
                min_thickness = std::min(min_thickness, state_->ice_thickness[i + j]);
                max_thickness = std::max(max_thickness, state_->ice_thickness[i + j]);
            }
        }
        
        // Handle remainder with scalar code
        for (; i < num_faces_; ++i) {
            double lat_rad = face_latitudes_[i] * M_PI / 180.0;
            double sin_lat = std::sin(lat_rad);
            double sin_lat_sq = sin_lat * sin_lat;  // Already positive after squaring
            
            double thickness = equatorial_thickness + thickness_diff * sin_lat_sq;
            state_->ice_thickness[i] = thickness;
            state_->accumulated_mass[i] = thickness * constants::ICE_DENSITY;
            
            min_thickness = std::min(min_thickness, thickness);
            max_thickness = std::max(max_thickness, thickness);
        }
    }
#elif defined(__AVX2__)
    // AVX2 implementation (no native sin, so hybrid approach)
    {
        const __m256d ice_dens = _m256_set1_pd(constants::ICE_DENSITY);
        
        #pragma omp parallel for reduction(min:min_thickness) reduction(max:max_thickness)
        for (size_t i = 0; i < num_faces_; i += 4) {
            // Process 4 faces at a time
            size_t batch_size = std::min(size_t(4), num_faces_ - i);
            
            alignas(32) double thickness_vals[4];
            alignas(32) double mass_vals[4];
            
            // Calculate thickness for batch (scalar trig functions)
            for (size_t j = 0; j < batch_size; ++j) {
                double lat_rad = face_latitudes_[i + j] * M_PI / 180.0;
                double sin_lat = std::sin(lat_rad);
                double sin_lat_sq = sin_lat * sin_lat;
                
                thickness_vals[j] = equatorial_thickness + thickness_diff * sin_lat_sq;
                mass_vals[j] = thickness_vals[j] * constants::ICE_DENSITY;
                
                min_thickness = std::min(min_thickness, thickness_vals[j]);
                max_thickness = std::max(max_thickness, thickness_vals[j]);
            }
            
            // Vectorized store
            if (batch_size == 4) {
                __m256d thickness = _mm256_load_pd(thickness_vals);
                __m256d mass = _mm256_load_pd(mass_vals);
                _mm256_storeu_pd(&state_->ice_thickness[i], thickness);
                _mm256_storeu_pd(&state_->accumulated_mass[i], mass);
            } else {
                // Handle partial batch
                for (size_t j = 0; j < batch_size; ++j) {
                    state_->ice_thickness[i + j] = thickness_vals[j];
                    state_->accumulated_mass[i + j] = mass_vals[j];
                }
            }
        }
    }
#else
    // Scalar implementation with OpenMP
    {
        #pragma omp parallel for reduction(min:min_thickness) reduction(max:max_thickness)
        for (size_t i = 0; i < num_faces_; ++i) {
            // Convert latitude to radians
            double lat_rad = face_latitudes_[i] * M_PI / 180.0;
            
            // Calculate latitude-dependent thickness
            // Using sin^2(lat) for smooth transition from equator to pole
            double sin_lat = std::sin(lat_rad);
            double sin_lat_sq = sin_lat * sin_lat;  // Always positive
            
            // Interpolate between equatorial and polar thickness
            double thickness = equatorial_thickness + thickness_diff * sin_lat_sq;
            
            // Set ice thickness and accumulated mass
            state_->ice_thickness[i] = thickness;
            state_->accumulated_mass[i] = thickness * constants::ICE_DENSITY;
            
            // Track statistics
            min_thickness = std::min(min_thickness, thickness);
            max_thickness = std::max(max_thickness, thickness);
        }
    }
#endif
    
    // Calculate distribution statistics
    double total_mass = 0.0;
    double equatorial_mass = 0.0;
    double mid_latitude_mass = 0.0;
    double polar_mass = 0.0;
    
    int equatorial_faces = 0;
    int mid_latitude_faces = 0;
    int polar_faces = 0;
    
    double equatorial_avg_thickness = 0.0;
    double mid_latitude_avg_thickness = 0.0;
    double polar_avg_thickness = 0.0;
    
    // Analyze distribution by latitude zones
    #pragma omp parallel for reduction(+:total_mass,equatorial_mass,mid_latitude_mass,polar_mass,\
                                        equatorial_faces,mid_latitude_faces,polar_faces,\
                                        equatorial_avg_thickness,mid_latitude_avg_thickness,polar_avg_thickness)
    for (size_t i = 0; i < num_faces_; ++i) {
        double face_mass = state_->accumulated_mass[i] * face_areas_[i];
        total_mass += face_mass;
        
        double abs_lat = std::abs(face_latitudes_[i]);
        
        if (abs_lat < 30.0) {
            // Equatorial zone: |latitude| < 30°
            equatorial_mass += face_mass;
            equatorial_avg_thickness += state_->ice_thickness[i];
            equatorial_faces++;
        } else if (abs_lat < 60.0) {
            // Mid-latitude zone: 30° <= |latitude| < 60°
            mid_latitude_mass += face_mass;
            mid_latitude_avg_thickness += state_->ice_thickness[i];
            mid_latitude_faces++;
        } else {
            // Polar zone: |latitude| >= 60°
            polar_mass += face_mass;
            polar_avg_thickness += state_->ice_thickness[i];
            polar_faces++;
        }
    }
    
    // Calculate averages
    if (equatorial_faces > 0) {
        equatorial_avg_thickness /= equatorial_faces;
    }
    if (mid_latitude_faces > 0) {
        mid_latitude_avg_thickness /= mid_latitude_faces;
    }
    if (polar_faces > 0) {
        polar_avg_thickness /= polar_faces;
    }
    
    // Calculate hemisphere asymmetry
    double north_mass = 0.0;
    double south_mass = 0.0;
    
    for (size_t i = 0; i < num_faces_; ++i) {
        double face_mass = state_->accumulated_mass[i] * face_areas_[i];
        if (face_latitudes_[i] > 0) {
            north_mass += face_mass;
        } else {
            south_mass += face_mass;
        }
    }
    
    // Report distribution statistics
    std::cout << "\n========================================" << std::endl;
    std::cout << "Latitudinal Ice Distribution Initialized" << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::cout << "\nInput Parameters:" << std::endl;
    std::cout << "  Equatorial thickness: " << std::fixed << std::setprecision(1) 
              << equatorial_thickness * 1e6 << " μm" << std::endl;
    std::cout << "  Polar thickness: " << polar_thickness * 1e6 << " μm" << std::endl;
    
    std::cout << "\nActual Distribution:" << std::endl;
    std::cout << "  Minimum thickness: " << min_thickness * 1e6 << " μm" << std::endl;
    std::cout << "  Maximum thickness: " << max_thickness * 1e6 << " μm" << std::endl;
    std::cout << "  Ratio (max/min): " << std::fixed << std::setprecision(2) 
              << (max_thickness / std::max(min_thickness, 1e-20)) << std::endl;
    
    std::cout << "\nLatitude Zone Statistics:" << std::endl;
    std::cout << "  Equatorial (|lat| < 30°):" << std::endl;
    std::cout << "    Faces: " << equatorial_faces << std::endl;
    std::cout << "    Avg thickness: " << std::fixed << std::setprecision(1) 
              << equatorial_avg_thickness * 1e6 << " μm" << std::endl;
    std::cout << "    Mass fraction: " << std::fixed << std::setprecision(1) 
              << (equatorial_mass / total_mass * 100.0) << "%" << std::endl;
    
    std::cout << "  Mid-latitude (30° ≤ |lat| < 60°):" << std::endl;
    std::cout << "    Faces: " << mid_latitude_faces << std::endl;
    std::cout << "    Avg thickness: " << mid_latitude_avg_thickness * 1e6 << " μm" << std::endl;
    std::cout << "    Mass fraction: " << (mid_latitude_mass / total_mass * 100.0) << "%" << std::endl;
    
    std::cout << "  Polar (|lat| ≥ 60°):" << std::endl;
    std::cout << "    Faces: " << polar_faces << std::endl;
    std::cout << "    Avg thickness: " << polar_avg_thickness * 1e6 << " μm" << std::endl;
    std::cout << "    Mass fraction: " << (polar_mass / total_mass * 100.0) << "%" << std::endl;
    
    std::cout << "\nHemisphere Balance:" << std::endl;
    std::cout << "  Northern hemisphere: " << std::scientific << std::setprecision(3) 
              << north_mass << " kg (" << std::fixed << std::setprecision(1) 
              << (north_mass / total_mass * 100.0) << "%)" << std::endl;
    std::cout << "  Southern hemisphere: " << std::scientific << std::setprecision(3) 
              << south_mass << " kg (" << std::fixed << std::setprecision(1) 
              << (south_mass / total_mass * 100.0) << "%)" << std::endl;
    std::cout << "  N/S ratio: " << std::fixed << std::setprecision(3) 
              << (north_mass / std::max(south_mass, 1e-20)) << std::endl;
    
    std::cout << "\nGlobal Statistics:" << std::endl;
    std::cout << "  Total ice mass: " << std::scientific << std::setprecision(4) 
              << total_mass << " kg" << std::endl;
    
    if (total_mass_before > 0.0) {
        std::cout << "  Previous mass: " << total_mass_before << " kg" << std::endl;
        std::cout << "  Mass change: " << (total_mass - total_mass_before) << " kg" << std::endl;
    }
    
    // Convert to equivalent global layer thickness
    double global_surface_area = 4.0 * M_PI * constants::EUROPA_RADIUS * constants::EUROPA_RADIUS;
    double equivalent_thickness = total_mass / (constants::ICE_DENSITY * global_surface_area);
    std::cout << "  Equivalent uniform thickness: " << std::fixed << std::setprecision(2) 
              << equivalent_thickness * 1e6 << " μm" << std::endl;
    
    std::cout << "========================================\n" << std::endl;
    
    // Optional: Store distribution parameters for later reference
    initial_equatorial_thickness_ = equatorial_thickness;
    initial_polar_thickness_ = polar_thickness;
    
    // Update global mass tracking
    total_mass_sublimated_ = 0.0;
    total_mass_deposited_ = 0.0;
    total_mass_escaped_ = 0.0;
}

void ExosphereModel::setInitialIceDistribution(const std::vector<double>& initial_mass) {
    // Input validation
    if (initial_mass.size() != num_faces_) {
        throw std::invalid_argument(
            "Initial mass array size (" + std::to_string(initial_mass.size()) + 
            ") does not match number of faces (" + std::to_string(num_faces_) + ")");
    }
    
    // Check for any negative values
    for (size_t i = 0; i < num_faces_; ++i) {
        if (initial_mass[i] < 0.0) {
            throw std::invalid_argument(
                "Negative mass value at face " + std::to_string(i) + 
                ": " + std::to_string(initial_mass[i]) + " kg/m²");
        }
        if (!std::isfinite(initial_mass[i])) {
            throw std::invalid_argument(
                "Non-finite mass value at face " + std::to_string(i));
        }
    }
    
    // Store previous total mass for comparison
    double total_mass_before = getTotalIceMass();
    
    // Statistics variables
    double min_mass = std::numeric_limits<double>::max();
    double max_mass = 0.0;
    double total_mass_new = 0.0;
    size_t zero_mass_faces = 0;
    size_t nonzero_mass_faces = 0;
    
    // Copy data with vectorization where possible
#ifdef __AVX512F__
    {
        const __m512d ice_density_inv = _mm512_set1_pd(1.0 / constants::ICE_DENSITY);
        const __m512d zero = _mm512_setzero_pd();
        
        size_t i = 0;
        for (; i + 8 <= num_faces_; i += 8) {
            // Load mass values
            __m512d mass = _mm512_loadu_pd(&initial_mass[i]);
            
            // Calculate ice thickness = mass / density
            __m512d thickness = _mm512_mul_pd(mass, ice_density_inv);
            
            // Store values
            _mm512_storeu_pd(&state_->accumulated_mass[i], mass);
            _mm512_storeu_pd(&state_->ice_thickness[i], thickness);
            
            // Count zero/nonzero faces (using mask operations)
            __mmask8 zero_mask = _mm512_cmp_pd_mask(mass, zero, _CMP_EQ_OQ);
            zero_mass_faces += __builtin_popcount(zero_mask);
            nonzero_mass_faces += __builtin_popcount(~zero_mask & 0xFF);
            
            // Update statistics (scalar for min/max)
            for (size_t j = 0; j < 8 && (i + j) < num_faces_; ++j) {
                double m = initial_mass[i + j];
                if (m > 0.0) {
                    min_mass = std::min(min_mass, m);
                    max_mass = std::max(max_mass, m);
                }
                total_mass_new += m * face_areas_[i + j];
            }
        }
        
        // Handle remainder
        for (; i < num_faces_; ++i) {
            state_->accumulated_mass[i] = initial_mass[i];
            state_->ice_thickness[i] = initial_mass[i] / constants::ICE_DENSITY;
            
            if (initial_mass[i] == 0.0) {
                zero_mass_faces++;
            } else {
                nonzero_mass_faces++;
                min_mass = std::min(min_mass, initial_mass[i]);
                max_mass = std::max(max_mass, initial_mass[i]);
            }
            total_mass_new += initial_mass[i] * face_areas_[i];
        }
    }
#elif defined(__AVX2__)
    {
        const __m256d ice_density_inv = _mm256_set1_pd(1.0 / constants::ICE_DENSITY);
        const __m256d zero = _mm256_setzero_pd();
        
        size_t i = 0;
        for (; i + 4 <= num_faces_; i += 4) {
            // Load mass values
            __m256d mass = _mm256_loadu_pd(&initial_mass[i]);
            
            // Calculate ice thickness
            __m256d thickness = _mm256_mul_pd(mass, ice_density_inv);
            
            // Store values
            _mm256_storeu_pd(&state_->accumulated_mass[i], mass);
            _mm256_storeu_pd(&state_->ice_thickness[i], thickness);
            
            // Update statistics (scalar)
            for (size_t j = 0; j < 4 && (i + j) < num_faces_; ++j) {
                double m = initial_mass[i + j];
                if (m == 0.0) {
                    zero_mass_faces++;
                } else {
                    nonzero_mass_faces++;
                    min_mass = std::min(min_mass, m);
                    max_mass = std::max(max_mass, m);
                }
                total_mass_new += m * face_areas_[i + j];
            }
        }
        
        // Handle remainder
        for (; i < num_faces_; ++i) {
            state_->accumulated_mass[i] = initial_mass[i];
            state_->ice_thickness[i] = initial_mass[i] / constants::ICE_DENSITY;
            
            if (initial_mass[i] == 0.0) {
                zero_mass_faces++;
            } else {
                nonzero_mass_faces++;
                min_mass = std::min(min_mass, initial_mass[i]);
                max_mass = std::max(max_mass, initial_mass[i]);
            }
            total_mass_new += initial_mass[i] * face_areas_[i];
        }
    }
#else
    // Scalar implementation with OpenMP
    {
        #pragma omp parallel for reduction(+:total_mass_new,zero_mass_faces,nonzero_mass_faces) \
                                reduction(min:min_mass) reduction(max:max_mass)
        for (size_t i = 0; i < num_faces_; ++i) {
            // Set mass and thickness
            state_->accumulated_mass[i] = initial_mass[i];
            state_->ice_thickness[i] = initial_mass[i] / constants::ICE_DENSITY;
            
            // Update statistics
            if (initial_mass[i] == 0.0) {
                zero_mass_faces++;
            } else {
                nonzero_mass_faces++;
                min_mass = std::min(min_mass, initial_mass[i]);
                max_mass = std::max(max_mass, initial_mass[i]);
            }
            
            total_mass_new += initial_mass[i] * face_areas_[i];
        }
    }
#endif
    
    // Calculate spatial distribution statistics
    double mean_mass = 0.0;
    double std_dev_mass = 0.0;
    double median_mass = 0.0;
    
    // Calculate mean (only for non-zero faces)
    if (nonzero_mass_faces > 0) {
        double sum_nonzero = 0.0;
        for (size_t i = 0; i < num_faces_; ++i) {
            if (initial_mass[i] > 0.0) {
                sum_nonzero += initial_mass[i];
            }
        }
        mean_mass = sum_nonzero / nonzero_mass_faces;
        
        // Calculate standard deviation
        double sum_sq_diff = 0.0;
        for (size_t i = 0; i < num_faces_; ++i) {
            if (initial_mass[i] > 0.0) {
                double diff = initial_mass[i] - mean_mass;
                sum_sq_diff += diff * diff;
            }
        }
        std_dev_mass = std::sqrt(sum_sq_diff / nonzero_mass_faces);
        
        // Calculate median (approximate for large arrays)
        std::vector<double> nonzero_masses;
        nonzero_masses.reserve(nonzero_mass_faces);
        for (size_t i = 0; i < num_faces_; ++i) {
            if (initial_mass[i] > 0.0) {
                nonzero_masses.push_back(initial_mass[i]);
            }
        }
        if (!nonzero_masses.empty()) {
            size_t median_idx = nonzero_masses.size() / 2;
            std::nth_element(nonzero_masses.begin(), 
                           nonzero_masses.begin() + median_idx, 
                           nonzero_masses.end());
            median_mass = nonzero_masses[median_idx];
        }
    }
    
    // If coordinates are available, calculate latitude distribution
    double equatorial_mass = 0.0;
    double mid_latitude_mass = 0.0;
    double polar_mass = 0.0;
    bool has_coordinates = !face_latitudes_.empty();
    
    if (has_coordinates) {
        for (size_t i = 0; i < num_faces_; ++i) {
            double face_total_mass = initial_mass[i] * face_areas_[i];
            double abs_lat = std::abs(face_latitudes_[i]);
            
            if (abs_lat < 30.0) {
                equatorial_mass += face_total_mass;
            } else if (abs_lat < 60.0) {
                mid_latitude_mass += face_total_mass;
            } else {
                polar_mass += face_total_mass;
            }
        }
    }
    
    // Calculate ice thickness statistics
    double min_thickness = (min_mass == std::numeric_limits<double>::max()) ? 
                          0.0 : min_mass / constants::ICE_DENSITY;
    double max_thickness = max_mass / constants::ICE_DENSITY;
    double mean_thickness = mean_mass / constants::ICE_DENSITY;
    
    // Report comprehensive statistics
    std::cout << "\n========================================" << std::endl;
    std::cout << "Initial Ice Distribution Set" << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::cout << "\nDistribution Statistics:" << std::endl;
    std::cout << "  Total faces: " << num_faces_ << std::endl;
    std::cout << "  Faces with ice: " << nonzero_mass_faces 
              << " (" << std::fixed << std::setprecision(1) 
              << (100.0 * nonzero_mass_faces / num_faces_) << "%)" << std::endl;
    std::cout << "  Faces without ice: " << zero_mass_faces 
              << " (" << (100.0 * zero_mass_faces / num_faces_) << "%)" << std::endl;
    
    if (nonzero_mass_faces > 0) {
        std::cout << "\nIce Coverage (non-zero faces only):" << std::endl;
        std::cout << "  Accumulated Mass (kg/m²):" << std::endl;
        std::cout << "    Minimum: " << std::scientific << std::setprecision(3) 
                  << min_mass << std::endl;
        std::cout << "    Maximum: " << max_mass << std::endl;
        std::cout << "    Mean: " << mean_mass << std::endl;
        std::cout << "    Median: " << median_mass << std::endl;
        std::cout << "    Std Dev: " << std_dev_mass << std::endl;
        std::cout << "    Ratio (max/min): " << std::fixed << std::setprecision(2) 
                  << (max_mass / std::max(min_mass, 1e-20)) << std::endl;
        
        std::cout << "\n  Ice Thickness:" << std::endl;
        std::cout << "    Minimum: " << std::fixed << std::setprecision(1) 
                  << min_thickness * 1e6 << " μm" << std::endl;
        std::cout << "    Maximum: " << max_thickness * 1e6 << " μm" << std::endl;
        std::cout << "    Mean: " << mean_thickness * 1e6 << " μm" << std::endl;
    }
    
    if (has_coordinates && total_mass_new > 0.0) {
        std::cout << "\nLatitudinal Distribution:" << std::endl;
        std::cout << "  Equatorial (|lat| < 30°): " << std::fixed << std::setprecision(1)
                  << (equatorial_mass / total_mass_new * 100.0) << "%" << std::endl;
        std::cout << "  Mid-latitude (30° ≤ |lat| < 60°): " 
                  << (mid_latitude_mass / total_mass_new * 100.0) << "%" << std::endl;
        std::cout << "  Polar (|lat| ≥ 60°): " 
                  << (polar_mass / total_mass_new * 100.0) << "%" << std::endl;
    }
    
    std::cout << "\nGlobal Statistics:" << std::endl;
    std::cout << "  Total ice mass: " << std::scientific << std::setprecision(4) 
              << total_mass_new << " kg" << std::endl;
    
    if (total_mass_before > 0.0) {
        double mass_change = total_mass_new - total_mass_before;
        std::cout << "  Previous mass: " << total_mass_before << " kg" << std::endl;
        std::cout << "  Mass change: " << mass_change << " kg ";
        
        if (std::abs(mass_change) > 1e-10) {
            std::cout << "(" << std::showpos << std::fixed << std::setprecision(1) 
                      << (mass_change / total_mass_before * 100.0) << "%)" << std::noshowpos;
        }
        std::cout << std::endl;
    }
    
    // Calculate equivalent uniform thickness
    double global_surface_area = 4.0 * M_PI * constants::EUROPA_RADIUS * constants::EUROPA_RADIUS;
    double equivalent_thickness = total_mass_new / (constants::ICE_DENSITY * global_surface_area);
    std::cout << "  Equivalent uniform thickness: " << std::fixed << std::setprecision(2) 
              << equivalent_thickness * 1e6 << " μm" << std::endl;
    
    // Surface area coverage
    double ice_covered_area = 0.0;
    for (size_t i = 0; i < num_faces_; ++i) {
        if (initial_mass[i] > 0.0) {
            ice_covered_area += face_areas_[i];
        }
    }
    std::cout << "  Ice-covered surface area: " << std::scientific << std::setprecision(3)
              << ice_covered_area / 1e6 << " km² (" << std::fixed << std::setprecision(1)
              << (ice_covered_area / global_surface_area * 100.0) << "%)" << std::endl;
    
    std::cout << "========================================\n" << std::endl;
    
    // Reset mass tracking counters
    total_mass_sublimated_ = 0.0;
    total_mass_deposited_ = 0.0;
    total_mass_escaped_ = 0.0;
}

void ExosphereModel::setInitialIceDistribution(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open ice distribution file: " + filename);
    }
    
    std::vector<double> initial_mass;
    initial_mass.reserve(num_faces_);
    
    std::string line;
    size_t line_num = 0;
    
    while (std::getline(file, line)) {
        line_num++;
        
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        // Parse the mass value
        try {
            double mass = std::stod(line);
            initial_mass.push_back(mass);
        } catch (const std::exception& e) {
            throw std::runtime_error("Error parsing line " + std::to_string(line_num) + 
                                   " in file " + filename + ": " + e.what());
        }
        
        // Check if we have too many values
        if (initial_mass.size() > num_faces_) {
            throw std::runtime_error("Too many values in ice distribution file. Expected " +
                                   std::to_string(num_faces_) + " but file has more.");
        }
    }
    
    // Check if we have the right number of values
    if (initial_mass.size() != num_faces_) {
        throw std::runtime_error("Ice distribution file has " + 
                               std::to_string(initial_mass.size()) + 
                               " values but mesh has " + std::to_string(num_faces_) + " faces.");
    }
    
    file.close();
    
    std::cout << "Loading ice distribution from file: " << filename << std::endl;
    setInitialIceDistribution(initial_mass);
}

// Helper function to export current distribution to file
void ExosphereModel::exportIceDistribution(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot create ice distribution file: " + filename);
    }
    
    file << "# Europa Exosphere Model - Ice Distribution\n";
    file << "# Format: accumulated_mass (kg/m²) per face\n";
    file << "# Number of faces: " << num_faces_ << "\n";
    file << "# Total ice mass: " << getTotalIceMass() << " kg\n";
    file << std::scientific << std::setprecision(12);
    
    for (size_t i = 0; i < num_faces_; ++i) {
        file << state_->accumulated_mass[i] << "\n";
    }
    
    file.close();
    std::cout << "Ice distribution exported to: " << filename << std::endl;
}

void ExosphereModel::evolveTimeStep(double dt) {
    // Build emission vector (total mass rate from each face)
    std::vector<double> emission_vec(num_faces_);
    for (size_t i = 0; i < num_faces_; ++i) {
        emission_vec[i] = state_->emission_flux[i] * face_areas_[i];
    }

    // Use matrix-free transport product (O(N) memory, no sparse matrix needed)
    std::vector<double> deposition_vec;
    if (!last_temperatures_.empty()) {
        computeTransportProduct(last_temperatures_, emission_vec, deposition_vec);
    } else {
        // Fallback: use pre-built transport matrix if temperatures not stored
        if (!transport_matrix_valid_) {
            updateTransportMatrix();
        }
        Eigen::VectorXd em(num_faces_);
        for (size_t i = 0; i < num_faces_; ++i) em(i) = emission_vec[i];
        Eigen::VectorXd dep = transport_matrix_ * em;
        deposition_vec.resize(num_faces_);
        for (size_t i = 0; i < num_faces_; ++i) deposition_vec[i] = dep(i);
    }

    // Calculate net fluxes
    #pragma omp parallel for
    for (size_t i = 0; i < num_faces_; ++i) {
        state_->deposition_flux[i] = deposition_vec[i] / face_areas_[i];
        state_->net_flux[i] = state_->deposition_flux[i] - state_->emission_flux[i];
    }

    // Update accumulated mass using vectorized function
    #ifdef USE_AVX512
        updateMassAVX512(
            state_->accumulated_mass.data(),
            state_->ice_thickness.data(),
            state_->net_flux.data(),
            dt,
            num_faces_
        );
    #elif defined(USE_AVX2)
        updateMassAVX2(
            state_->accumulated_mass.data(),
            state_->ice_thickness.data(),
            state_->net_flux.data(),
            dt,
            num_faces_
        );
    #else
        // Scalar fallback with OpenMP
        #pragma omp parallel for
        for (size_t i = 0; i < num_faces_; ++i) {
            double new_mass = state_->accumulated_mass[i] + state_->net_flux[i] * dt;
            state_->accumulated_mass[i] = std::max(0.0, new_mass);
            state_->ice_thickness[i] = state_->accumulated_mass[i] / constants::ICE_DENSITY;
        }
    #endif

    // Calculate totals for mass balance checking
    double total_emission = 0.0;
    double total_deposition = 0.0;
    double total_escape = 0.0;

    #pragma omp parallel for reduction(+:total_emission,total_deposition,total_escape)
    for (size_t i = 0; i < num_faces_; ++i) {
        total_emission += emission_vec[i];
        total_deposition += deposition_vec[i];
        total_escape += state_->escape_flux[i] * face_areas_[i];
    }
    
    // Update global statistics
    total_mass_sublimated_ += total_emission * dt;
    total_mass_deposited_ += total_deposition * dt;
    total_mass_escaped_ += total_escape * dt;
    
    // Verify mass conservation
    double balance_error = std::abs(total_emission - total_deposition - total_escape) / 
                          std::max(total_emission, 1e-20);
    
    if (balance_error > 1e-3) {
        std::cout << "WARNING: Mass balance error = " << balance_error << std::endl;
    }
}

void ExosphereModel::evolveMultipleSteps(double dt, int n_steps) {
    for (int step = 0; step < n_steps; ++step) {
        evolveTimeStep(dt);
        
        if (step % 10 == 0) {
            std::cout << "Step " << step << "/" << n_steps << " completed" << std::endl;
        }
    }
}

void ExosphereModel::solveSteadyState(int max_iterations, double tolerance) {
    std::cout << "Solving for steady-state exosphere..." << std::endl;
    
    updateTransportMatrix();
    
    // Power iteration method
    Eigen::VectorXd flux_current(num_faces_);
    Eigen::VectorXd flux_next(num_faces_);
    
    for (size_t i = 0; i < num_faces_; ++i) {
        flux_current(i) = state_->emission_flux[i];
    }
    
    double error = 1.0;
    int iteration = 0;
    
    while (iteration < max_iterations && error > tolerance) {
        flux_next = transport_matrix_.transpose() * flux_current;
        error = (flux_next - flux_current).norm() / flux_current.norm();
        flux_current = flux_next;
        iteration++;
        
        if (iteration % 100 == 0) {
            std::cout << "  Iteration " << iteration << ", error = " << error << std::endl;
        }
    }
    
    for (size_t i = 0; i < num_faces_; ++i) {
        state_->net_flux[i] = flux_current(i) - state_->emission_flux[i];
    }
    
    std::cout << "Steady state " << (error <= tolerance ? "converged" : "did not converge")
              << " after " << iteration << " iterations (error = " << error << ")" << std::endl;
}

void ExosphereModel::solveMatrixEquation(double dt) {
    updateTransportMatrix();
    
    // FIXED: Properly handle sparse matrix operations
    // Build A = I - dt*F^T using proper sparse operations
    
    size_t n = num_faces_;
    
    // Create identity matrix
    Eigen::SparseMatrix<double> I(n, n);
    I.setIdentity();
    
    // Create dt*F^T
    Eigen::SparseMatrix<double> dtFT = dt * transport_matrix_.transpose();
    
    // Build A = I - dt*F^T (both are sparse, so this works)
    Eigen::SparseMatrix<double> A = I - dtFT;
    
    // Right-hand side
    Eigen::VectorXd b(n);
    for (size_t i = 0; i < n; ++i) {
        b(i) = state_->accumulated_mass[i] + dt * state_->emission_flux[i] * face_areas_[i];
    }
    
    // Solve using BiCGSTAB for better performance
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solver;
    solver.setMaxIterations(max_iterations_);
    solver.setTolerance(convergence_threshold_);
    solver.compute(A);
    
    if (solver.info() != Eigen::Success) {
        std::cerr << "Matrix decomposition failed!" << std::endl;
        return;
    }
    
    Eigen::VectorXd m_new = solver.solve(b);
    
    if (solver.info() != Eigen::Success) {
        std::cerr << "Matrix solve failed! Iterations: " << solver.iterations() 
                  << ", Error: " << solver.error() << std::endl;
        return;
    }
    
    // Update state
    for (size_t i = 0; i < n; ++i) {
        state_->net_flux[i] = (m_new(i) - state_->accumulated_mass[i]) / dt / face_areas_[i];
        state_->accumulated_mass[i] = m_new(i);
        state_->ice_thickness[i] = state_->accumulated_mass[i] / constants::ICE_DENSITY;
    }
    
    std::cout << "Matrix equation solved in " << solver.iterations() << " iterations" << std::endl;
}

void ExosphereModel::solveImplicitEuler(double dt) {
    // Implicit Euler: m^(n+1) = m^n + dt * F^T * emission^(n+1)
    solveMatrixEquation(dt);
}

void ExosphereModel::calculateMassBalance() {
    double total_emission = 0.0;
    double total_deposition = 0.0;
    double total_escape = 0.0;
    double total_accumulated = 0.0;
    
    #pragma omp parallel for reduction(+:total_emission,total_deposition,total_escape,total_accumulated)
    for (size_t i = 0; i < num_faces_; ++i) {
        total_emission += state_->emission_flux[i] * face_areas_[i];
        total_deposition += state_->deposition_flux[i] * face_areas_[i];
        total_escape += state_->escape_flux[i] * face_areas_[i];
        total_accumulated += state_->accumulated_mass[i] * face_areas_[i];
    }
    
    std::cout << "\nMass Balance Summary:" << std::endl;
    std::cout << "  Current emission rate: " << total_emission << " kg/s" << std::endl;
    std::cout << "  Current deposition rate: " << total_deposition << " kg/s" << std::endl;
    std::cout << "  Current escape rate: " << total_escape << " kg/s" << std::endl;
    std::cout << "  Total accumulated mass: " << total_accumulated << " kg" << std::endl;
    std::cout << "  Balance error: " << std::abs(total_emission - total_deposition - total_escape) 
              << " kg/s" << std::endl;
}

void ExosphereModel::calculateLatitudinalDistribution(std::vector<double>& lat_bins,
                                                     std::vector<double>& mass_bins,
                                                     int n_bins) {
    lat_bins.resize(n_bins);
    mass_bins.resize(n_bins, 0.0);
    
    double lat_min = -90.0;
    double lat_max = 90.0;
    double dlat = (lat_max - lat_min) / n_bins;
    
    for (int i = 0; i < n_bins; ++i) {
        lat_bins[i] = lat_min + (i + 0.5) * dlat;
    }
    
    // Accumulate mass by latitude bin
    for (size_t i = 0; i < num_faces_; ++i) {
        int bin = static_cast<int>((face_latitudes_[i] - lat_min) / dlat);
        bin = std::max(0, std::min(n_bins - 1, bin));
        mass_bins[bin] += state_->accumulated_mass[i] * face_areas_[i];
    }
}

void ExosphereModel::exportState(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    
    file << std::setprecision(12);
    file << "# Europa Exosphere State\n";
    file << "# Face_ID, Lat(deg), Lon(deg), Vapor_Pressure(Pa), Vapor_Density(kg/m3), ";
    file << "Thermal_Velocity(m/s), Emission_Flux(kg/m2/s), Deposition_Flux(kg/m2/s), ";
    file << "Net_Flux(kg/m2/s), Accumulated_Mass(kg/m2), Ice_Thickness(m), Escape_Flux(kg/m2/s)\n";
    
    for (size_t i = 0; i < num_faces_; ++i) {
        file << i << ", "
             << face_latitudes_[i] << ", "
             << face_longitudes_[i] << ", "
             << state_->vapor_pressure[i] << ", "
             << state_->vapor_density[i] << ", "
             << state_->thermal_velocity[i] << ", "
             << state_->emission_flux[i] << ", "
             << state_->deposition_flux[i] << ", "
             << state_->net_flux[i] << ", "
             << state_->accumulated_mass[i] << ", "
             << state_->ice_thickness[i] << ", "
             << state_->escape_flux[i] << "\n";
    }
    
    file.close();
    std::cout << "Exosphere state exported to: " << filename << std::endl;
}

void ExosphereModel::exportTransportMatrix(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    
    file << "# Transport Matrix (sparse format)\n";
    file << "# Row(to), Col(from), Value\n";
    
    int count = 0;
    for (int k = 0; k < transport_matrix_.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(transport_matrix_, k); it; ++it) {
            if (it.value() > 1e-12) {
                file << it.row() << ", " << it.col() << ", " 
                     << std::scientific << std::setprecision(6) << it.value() << "\n";
                count++;
            }
        }
    }
    
    file.close();
    std::cout << "Transport matrix exported (" << count << " non-zeros) to: " << filename << std::endl;
}

void ExosphereModel::exportStatistics(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    
    file << "# Europa Exosphere Simulation Statistics\n";
    file << "# Generated at simulation completion\n\n";
    
    file << "Global Statistics:\n";
    file << "  Total mass sublimated: " << total_mass_sublimated_ << " kg\n";
    file << "  Total mass deposited: " << total_mass_deposited_ << " kg\n";
    file << "  Total mass escaped: " << total_mass_escaped_ << " kg\n";
    file << "  Net mass change: " << (total_mass_deposited_ - total_mass_sublimated_) << " kg\n";
    file << "  Escape fraction: " << (total_mass_escaped_ / std::max(total_mass_sublimated_, 1e-20)) << "\n\n";
    
    // Calculate regional statistics
    double polar_mass = 0.0, equatorial_mass = 0.0, mid_lat_mass = 0.0;
    double max_thickness = 0.0;
    size_t max_thickness_face = 0;
    
    for (size_t i = 0; i < num_faces_; ++i) {
        double lat = face_latitudes_[i];
        double mass = state_->accumulated_mass[i] * face_areas_[i];
        
        if (std::abs(lat) > 60.0) {
            polar_mass += mass;
        } else if (std::abs(lat) < 30.0) {
            equatorial_mass += mass;
        } else {
            mid_lat_mass += mass;
        }
        
        if (state_->ice_thickness[i] > max_thickness) {
            max_thickness = state_->ice_thickness[i];
            max_thickness_face = i;
        }
    }
    
    file << "Regional Distribution:\n";
    file << "  Polar mass (|lat| > 60°): " << polar_mass << " kg\n";
    file << "  Mid-latitude mass (30° < |lat| < 60°): " << mid_lat_mass << " kg\n";
    file << "  Equatorial mass (|lat| < 30°): " << equatorial_mass << " kg\n";
    file << "  Polar/Equatorial ratio: " << (polar_mass / std::max(equatorial_mass, 1e-20)) << "\n\n";
    
    file << "Ice Thickness:\n";
    file << "  Maximum thickness: " << max_thickness << " m\n";
    file << "  Location: face " << max_thickness_face 
         << " (lat=" << face_latitudes_[max_thickness_face] 
         << "°, lon=" << face_longitudes_[max_thickness_face] << "°)\n";
    
    file.close();
    std::cout << "Statistics exported to: " << filename << std::endl;
}

// Add a helper for radial normalization (Gauss-Legendre n=32)
namespace {
    inline double radialKernelNorm(double T) {
        if (T <= 0.0) return 1.0;
        constexpr int N = 32;
        static const double x[N] = {
            -0.9972638618,-0.9856115115,-0.9647622556,-0.9349060759,-0.8963211558,-0.8493676137,
            -0.7944837959,-0.7321821187,-0.6630442669,-0.5877157572,-0.5068999089,-0.4213512761,
            -0.3318686023,-0.2392873623,-0.1444719616,-0.0483076657, 0.0483076657, 0.1444719616,
             0.2392873623, 0.3318686023, 0.4213512761, 0.5068999089, 0.5877157572, 0.6630442669,
             0.7321821187, 0.7944837959, 0.8493676137, 0.8963211558, 0.9349060759, 0.9647622556,
             0.9856115115, 0.9972638618
        };
        static const double w[N] = {
            0.0070186100,0.0162743947,0.0253920653,0.0342738629,0.0428358980,0.0509980593,
            0.0586840935,0.0658222228,0.0723457941,0.0781938958,0.0833119242,0.0876520930,
            0.0911738787,0.0938443991,0.0956387201,0.0965400885,0.0965400885,0.0956387201,
            0.0938443991,0.0911738787,0.0876520930,0.0833119242,0.0781938958,0.0723457941,
            0.0658222228,0.0586840935,0.0509980593,0.0428358980,0.0342738629,0.0253920653,
            0.0162743947,0.0070186100
        };
        const double R = constants::EUROPA_RADIUS;
        double sigma2 = constants::BOLTZMANN_K * T / constants::WATER_MOLECULAR_MASS;
        if (sigma2 <= 0.0) return 1.0;
        double norm = 0.0;
        // θ in [0,π]
        for (int k=0;k<N;++k) {
            double t = 0.5*(x[k]+1.0)*M_PI;      // map [-1,1] -> [0,π]
            double d = R * t;
            double xi = constants::EUROPA_GRAVITY * d / (2.0 * sigma2);
            if (xi > 60.0) continue;
            double pref = constants::EUROPA_GRAVITY * constants::EUROPA_GRAVITY * d /
                          (4.0 * sigma2 * sigma2);
            double K = pref * besselK0(xi);
            if (K <= 0.0) continue;
            norm += w[k] * K * std::sin(t); // angular part
        }
        // Multiply by 2πR^2 * 0.5*π (Jacobian of mapping) = π * R^2
        norm *= M_PI * constants::EUROPA_RADIUS * constants::EUROPA_RADIUS;
        return (norm > 0.0) ? norm : 1.0;
    }
}

// Replace body of BallisticTransport::buildTransportMatrix with optimized no-sort version
Eigen::SparseMatrix<double> BallisticTransport::buildTransportMatrix(
    const std::vector<std::array<double, 3>>& face_centers,
    const std::vector<double>& face_areas,
    const std::vector<double>& temperatures,
    double threshold) const
{
    size_t n = face_centers.size();
    const double R = constants::EUROPA_RADIUS;

    // Precompute unit centers
    std::vector<std::array<double,3>> unit(n);
    for (size_t i=0;i<n;++i){
        double invr = 1.0 / std::sqrt(face_centers[i][0]*face_centers[i][0]+
                                      face_centers[i][1]*face_centers[i][1]+
                                      face_centers[i][2]*face_centers[i][2]);
        unit[i] = {face_centers[i][0]*invr, face_centers[i][1]*invr, face_centers[i][2]*invr};
    }

    // Mean angular size
    double mean_theta = 0.0;
    for (double A : face_areas) mean_theta += std::sqrt(std::max(A,0.0)/M_PI)/R;
    mean_theta /= std::max<size_t>(1, face_areas.size());

    // Shell factor (env configurable)
    double shell_factor = 2.5;
    if (const char* envp = std::getenv("EXO_SHELL_WIDTH_FACTOR")) {
        double v = std::atof(envp);
        if (v > 0.1 && v < 10.0) shell_factor = v;
    }
    const double dtheta_base = std::max(1e-6, shell_factor * mean_theta);
    const int max_shells = static_cast<int>(std::ceil(M_PI / dtheta_base)) + 2;

    bool disable_ring_corr = false;
    if (const char* envp = std::getenv("EXO_DISABLE_RING_CORR")) disable_ring_corr = (std::atoi(envp)!=0);

    auto p_density = [&](double d,double T){
        if (d<=0.0||T<=0.0) return 0.0;
        double sigma2 = constants::BOLTZMANN_K*T/constants::WATER_MOLECULAR_MASS;
        double xi = constants::EUROPA_GRAVITY*d/(2.0*sigma2);
        if (xi > 50.0) return 0.0;
        double sigma4 = sigma2*sigma2;
        return (constants::EUROPA_GRAVITY*constants::EUROPA_GRAVITY*d/(4.0*sigma4))*besselK0(xi);
    };

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(n*64);

    #pragma omp parallel
    {
        std::vector<Eigen::Triplet<double>> local;
        local.reserve(8192);

        std::vector<double> shell_disc_area(max_shells);
        std::vector<double> shell_cont_area(max_shells);
        std::vector<std::vector<std::pair<int,double>>> shell_entries(max_shells);

        #pragma omp for schedule(dynamic)
        for (size_t i=0;i<n;++i){
            double T_i = temperatures[i];
            if (T_i <= 0.0){
                local.emplace_back((int)i,(int)i,1.0);
                continue;
            }

            // Escape fraction (same approximation as before)
            double vth = std::sqrt(2.0 * constants::BOLTZMANN_K * T_i / constants::WATER_MOLECULAR_MASS);
            double ratio = constants::EUROPA_ESCAPE_VELOCITY / vth;
            double escape_frac = 0.0;
            if (ratio < 5.0) {
                escape_frac = std::exp(-ratio*ratio)*(1.0+ratio*ratio)/(2.0*ratio*ratio);
            }

            // Self-cap integral (reuse existing small 6-pt)
            auto selfCap = [&](double T, double A){
                if (T<=0.0||A<=0.0) return 0.0;
                const int N=6;
                static const double xG[6]={-0.9324695142031520,-0.6612093864662645,-0.2386191860831969,
                                            0.2386191860831969, 0.6612093864662645, 0.9324695142031520};
                static const double wG[6]={0.1713244923791704,0.3607615730481386,0.4679139345726910,
                                            0.4679139345726910,0.3607615730481386,0.1713244923791704};
                double theta_cap = std::sqrt(std::max(A,0.0)/(M_PI*R*R));
                theta_cap = std::min(theta_cap,0.25);
                double a=0,b=theta_cap;
                double integ=0;
                for(int k=0;k<N;++k){
                    double t=0.5*((b-a)*xG[k]+(b+a));
                    double d=R*t;
                    double pd=p_density(d,T);
                    double jac=0.5*(b-a);
                    integ += wG[k]*pd*(2.0*M_PI*R*R*std::sin(t))*jac;
                }
                return integ;
            };
            double self_mass = selfCap(T_i, face_areas[i]);

            // Reset shells
            std::fill(shell_disc_area.begin(), shell_disc_area.end(), 0.0);
            for (int s=0;s<max_shells;++s) shell_entries[s].clear();
            int used_shells = 0;

            const auto& ui = unit[i];

            // Bin directly (no sort)
            for (size_t j=0;j<n;++j){
                if (j==i) continue;
                const auto& uj = unit[j];
                double dotp = ui[0]*uj[0]+ui[1]*uj[1]+ui[2]*uj[2];
                dotp = std::max(-1.0,std::min(1.0,dotp));
                double theta = std::acos(dotp);
                int s = (int)std::floor(theta / dtheta_base);
                if (s >= max_shells) s = max_shells - 1;
                shell_disc_area[s] += face_areas[j];
                shell_entries[s].emplace_back((int)j, R * theta);
                if (s+1 > used_shells) used_shells = s+1;
            }

            for (int s=0;s<used_shells;++s){
                double theta_inner = s * dtheta_base;
                double theta_outer = (s+1)*dtheta_base;
                if (theta_outer > M_PI) theta_outer = M_PI;
                shell_cont_area[s] =
                    2.0 * M_PI * R * R * (std::cos(theta_inner) - std::cos(theta_outer));
            }

            double sum_off_raw = 0.0;
            std::vector<std::pair<int,double>> buf;
            buf.reserve(used_shells * 8);

            for (int s=0;s<used_shells;++s){
                if (shell_entries[s].empty()) continue;
                double correction = 1.0;
                if (!disable_ring_corr && shell_disc_area[s] > 0.0 && shell_cont_area[s] > 0.0) {
                    correction = shell_cont_area[s] / shell_disc_area[s];
                }
                for (auto& rec : shell_entries[s]) {
                    int jidx = rec.first;
                    double d = rec.second;
                    double pd = p_density(d, T_i);
                    if (pd <= 0.0) continue;
                    double w_raw = pd * face_areas[jidx] * correction;
                    if (w_raw < threshold) continue;
                    buf.emplace_back(jidx, w_raw);
                    sum_off_raw += w_raw;
                }
            }

            double total_raw = self_mass + sum_off_raw;
            double target = 1.0 - escape_frac;
            if (total_raw <= 0.0){
                local.emplace_back((int)i,(int)i,target);
                continue;
            }
            double scale = target / total_raw;
            local.emplace_back((int)i,(int)i,self_mass*scale);
            for (auto& pr : buf){
                local.emplace_back(pr.first,(int)i, pr.second*scale);
            }
        }

        #pragma omp critical
        {
            triplets.insert(triplets.end(), local.begin(), local.end());
        }
    }

    Eigen::SparseMatrix<double> M(n,n);
    M.setFromTriplets(triplets.begin(), triplets.end());
    M.makeCompressed();
    verifyMassConservation(M);
    return M;
}

// NEW: Optimized transport matrix build using distance cache
// This avoids repeated acos() computations by caching cos(theta) values
Eigen::SparseMatrix<double> BallisticTransport::buildTransportMatrixCached(
    const std::vector<std::array<double, 3>>& face_centers,
    const std::vector<double>& face_areas,
    const std::vector<double>& temperatures,
    double threshold) const
{
    size_t n = face_centers.size();
    const double R = body_radius_;

    auto build_start = std::chrono::high_resolution_clock::now();

    // Build distance cache if not already built or if size changed
    if (!optimized_cache_built_ || !optimized_cache_.isValid() ||
        optimized_cache_.numFaces() != n) {
        std::cout << "Building optimized distance cache..." << std::endl;
        optimized_cache_.build(face_centers);
        optimized_cache_built_ = true;
    }

    // Mean angular size for shell width
    double mean_theta = 0.0;
    for (double A : face_areas) mean_theta += std::sqrt(std::max(A, 0.0) / M_PI) / R;
    mean_theta /= std::max<size_t>(1, face_areas.size());

    // Shell factor (env configurable)
    double shell_factor = 2.5;
    if (const char* envp = std::getenv("EXO_SHELL_WIDTH_FACTOR")) {
        double v = std::atof(envp);
        if (v > 0.1 && v < 10.0) shell_factor = v;
    }
    const double dtheta_base = std::max(1e-6, shell_factor * mean_theta);
    const int max_shells = static_cast<int>(std::ceil(M_PI / dtheta_base)) + 2;

    bool disable_ring_corr = false;
    if (const char* envp = std::getenv("EXO_DISABLE_RING_CORR")) {
        disable_ring_corr = (std::atoi(envp) != 0);
    }

    // Kernel probability density function
    auto p_density = [&](double d, double T) {
        if (d <= 0.0 || T <= 0.0) return 0.0;
        double sigma2 = constants::BOLTZMANN_K * T / constants::WATER_MOLECULAR_MASS;
        double xi = constants::EUROPA_GRAVITY * d / (2.0 * sigma2);
        if (xi > 50.0) return 0.0;
        double sigma4 = sigma2 * sigma2;
        return (constants::EUROPA_GRAVITY * constants::EUROPA_GRAVITY * d / (4.0 * sigma4)) * besselK0(xi);
    };

    // Self-cap integral using 6-point Gauss-Legendre
    auto selfCap = [&](double T, double A) {
        if (T <= 0.0 || A <= 0.0) return 0.0;
        constexpr int NG = 6;
        static const double xG[NG] = {-0.9324695142031520, -0.6612093864662645, -0.2386191860831969,
                                       0.2386191860831969,  0.6612093864662645,  0.9324695142031520};
        static const double wG[NG] = {0.1713244923791704, 0.3607615730481386, 0.4679139345726910,
                                      0.4679139345726910, 0.3607615730481386, 0.1713244923791704};
        double theta_cap = std::sqrt(std::max(A, 0.0) / (M_PI * R * R));
        theta_cap = std::min(theta_cap, 0.25);
        double a = 0, b = theta_cap;
        double integ = 0;
        for (int k = 0; k < NG; ++k) {
            double t = 0.5 * ((b - a) * xG[k] + (b + a));
            double d = R * t;
            double pd = p_density(d, T);
            double jac = 0.5 * (b - a);
            integ += wG[k] * pd * (2.0 * M_PI * R * R * std::sin(t)) * jac;
        }
        return integ;
    };

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(n * 64);

    // Kernel cutoff optimization: compute max effective theta for temperature range
    // The kernel has xi = g*d/(2*sigma^2), and becomes negligible when xi > xi_cutoff
    // This means theta_max = 2*sigma^2*xi_cutoff / (g*R)
    // where sigma^2 = kT/m
    // Physics: K0(xi=15) = 1e-7 * K0(1), capturing >99.99% of transport probability
    double xi_cutoff = 15.0;  // Default: captures 99.99% of transport
    if (const char* envp = std::getenv("EXO_XI_CUTOFF")) {
        double v = std::atof(envp);
        if (v > 1.0 && v < 100.0) xi_cutoff = v;
    }

    // Compute cutoff angles for min and max temperatures
    double T_min = *std::min_element(temperatures.begin(), temperatures.end());
    double T_max = *std::max_element(temperatures.begin(), temperatures.end());
    T_min = std::max(T_min, 10.0);  // Floor at 10K to avoid division issues
    T_max = std::max(T_max, 10.0);

    // For a given T, theta_max = 2*k*T*xi_cutoff / (m*g*R)
    auto compute_theta_cutoff = [&](double T) {
        double sigma2 = constants::BOLTZMANN_K * T / constants::WATER_MOLECULAR_MASS;
        return 2.0 * sigma2 * xi_cutoff / (constants::EUROPA_GRAVITY * R);
    };

    double theta_cutoff_min = compute_theta_cutoff(T_min);
    double theta_cutoff_max = compute_theta_cutoff(T_max);

    // Precompute cos(theta_cutoff) for fast lookup comparison
    // Since cos is decreasing on [0,π], theta < theta_cutoff iff cos(theta) > cos(theta_cutoff)
    double cos_cutoff_min = std::cos(theta_cutoff_min);
    double cos_cutoff_max = std::cos(theta_cutoff_max);

    bool use_cutoff = true;
    if (const char* envp = std::getenv("EXO_NO_CUTOFF")) {
        use_cutoff = (std::atoi(envp) == 0);
    }

    std::cout << "Kernel cutoff: xi_max=" << xi_cutoff
              << ", theta_max=[" << (theta_cutoff_min * 180.0 / M_PI) << ", "
              << (theta_cutoff_max * 180.0 / M_PI) << "] deg"
              << ", T=[" << T_min << ", " << T_max << "] K"
              << ", cutoff " << (use_cutoff ? "ENABLED" : "DISABLED") << std::endl;

    // Statistics
    size_t total_acos_avoided = 0;
    size_t total_pairs = 0;
    size_t total_pairs_skipped = 0;

    #pragma omp parallel reduction(+:total_acos_avoided,total_pairs,total_pairs_skipped)
    {
        std::vector<Eigen::Triplet<double>> local;
        local.reserve(8192);

        std::vector<double> shell_disc_area(max_shells);
        std::vector<double> shell_cont_area(max_shells);
        std::vector<std::vector<std::pair<int, double>>> shell_entries(max_shells);

        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < n; ++i) {
            double T_i = temperatures[i];
            if (T_i <= 0.0) {
                local.emplace_back((int)i, (int)i, 1.0);
                continue;
            }

            // Escape fraction
            double vth = std::sqrt(2.0 * constants::BOLTZMANN_K * T_i / constants::WATER_MOLECULAR_MASS);
            double ratio = constants::EUROPA_ESCAPE_VELOCITY / vth;
            double escape_frac = 0.0;
            if (ratio < 5.0) {
                escape_frac = std::exp(-ratio * ratio) * (1.0 + ratio * ratio) / (2.0 * ratio * ratio);
            }

            double self_mass = selfCap(T_i, face_areas[i]);

            // Compute per-source cutoff angle based on source temperature
            // Higher T => larger thermal velocity => longer range transport
            double theta_cutoff_i = compute_theta_cutoff(T_i);
            float cos_cutoff_i = static_cast<float>(std::cos(theta_cutoff_i));

            // Reset shells
            std::fill(shell_disc_area.begin(), shell_disc_area.end(), 0.0);
            for (int s = 0; s < max_shells; ++s) shell_entries[s].clear();
            int used_shells = 0;

            // Use cached theta values - O(1) lookup, NO acos() computation!
            for (size_t j = 0; j < n; ++j) {
                if (j == i) continue;
                total_pairs++;

                double theta;
                if (use_cutoff) {
                    // CUTOFF OPTIMIZATION: Check cos(theta) first for early rejection
                    // cos(theta) > cos(theta_cutoff) means theta < theta_cutoff (cos decreasing)
                    float cos_theta_ij = optimized_cache_.getCosTheta(i, j);
                    if (cos_theta_ij < cos_cutoff_i) {
                        // Beyond cutoff - skip this pair
                        total_pairs_skipped++;
                        continue;
                    }
                    // Use cached theta for faces that pass cutoff
                    theta = optimized_cache_.getThetaCached(i, j);
                } else {
                    // No cutoff - directly use cached theta
                    theta = optimized_cache_.getThetaCached(i, j);
                }
                total_acos_avoided++;

                int s = static_cast<int>(std::floor(theta / dtheta_base));
                if (s >= max_shells) s = max_shells - 1;
                shell_disc_area[s] += face_areas[j];
                shell_entries[s].emplace_back(static_cast<int>(j), R * theta);
                if (s + 1 > used_shells) used_shells = s + 1;
            }

            // Compute continuous shell areas
            for (int s = 0; s < used_shells; ++s) {
                double theta_inner = s * dtheta_base;
                double theta_outer = (s + 1) * dtheta_base;
                if (theta_outer > M_PI) theta_outer = M_PI;
                shell_cont_area[s] =
                    2.0 * M_PI * R * R * (std::cos(theta_inner) - std::cos(theta_outer));
            }

            double sum_off_raw = 0.0;
            std::vector<std::pair<int, double>> buf;
            buf.reserve(used_shells * 8);

            for (int s = 0; s < used_shells; ++s) {
                if (shell_entries[s].empty()) continue;
                double correction = 1.0;
                if (!disable_ring_corr && shell_disc_area[s] > 0.0 && shell_cont_area[s] > 0.0) {
                    correction = shell_cont_area[s] / shell_disc_area[s];
                }
                for (auto& rec : shell_entries[s]) {
                    int jidx = rec.first;
                    double d = rec.second;
                    double pd = p_density(d, T_i);
                    if (pd <= 0.0) continue;
                    double w_raw = pd * face_areas[jidx] * correction;
                    if (w_raw < threshold) continue;
                    buf.emplace_back(jidx, w_raw);
                    sum_off_raw += w_raw;
                }
            }

            double total_raw = self_mass + sum_off_raw;
            double target = 1.0 - escape_frac;
            if (total_raw <= 0.0) {
                local.emplace_back((int)i, (int)i, target);
                continue;
            }
            double scale = target / total_raw;
            local.emplace_back((int)i, (int)i, self_mass * scale);
            for (auto& pr : buf) {
                local.emplace_back(pr.first, (int)i, pr.second * scale);
            }
        }

        #pragma omp critical
        {
            triplets.insert(triplets.end(), local.begin(), local.end());
        }
    }

    Eigen::SparseMatrix<double> M(n, n);
    M.setFromTriplets(triplets.begin(), triplets.end());
    M.makeCompressed();

    auto build_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(build_end - build_start);

    // Report statistics
    double skip_ratio = (total_pairs > 0) ?
        (100.0 * static_cast<double>(total_pairs_skipped) / static_cast<double>(total_pairs)) : 0.0;
    size_t pairs_evaluated = total_pairs - total_pairs_skipped;

    std::cout << "Transport matrix (cached) built: " << n << " faces, "
              << M.nonZeros() << " non-zeros, " << duration.count() << " ms" << std::endl;
    std::cout << "  Pair statistics: " << total_pairs << " total, "
              << pairs_evaluated << " evaluated, "
              << total_pairs_skipped << " skipped ("
              << std::fixed << std::setprecision(1) << skip_ratio << "% skip rate)" << std::endl;

    verifyMassConservation(M);
    return M;
}

// NEW: Spatially-indexed transport matrix build
// Uses spatial index for O(N*k) complexity where k = average number of neighbors
// This avoids the O(N²) neighbor iteration of the cached version
Eigen::SparseMatrix<double> BallisticTransport::buildTransportMatrixIndexed(
    const std::vector<std::array<double, 3>>& face_centers,
    const std::vector<double>& face_areas,
    const std::vector<double>& temperatures,
    double threshold) const
{
    size_t n = face_centers.size();
    const double R = body_radius_;

    auto build_start = std::chrono::high_resolution_clock::now();

    // Compute kernel cutoff parameters
    double xi_cutoff = 15.0;  // Default: captures 99.99% of transport
    if (const char* envp = std::getenv("EXO_XI_CUTOFF")) {
        double v = std::atof(envp);
        if (v > 1.0 && v < 100.0) xi_cutoff = v;
    }

    // Compute max theta for spatial index query radius
    double T_max = *std::max_element(temperatures.begin(), temperatures.end());
    T_max = std::max(T_max, 10.0);
    double sigma2_max = constants::BOLTZMANN_K * T_max / constants::WATER_MOLECULAR_MASS;
    double theta_max_global = 2.0 * sigma2_max * xi_cutoff / (constants::EUROPA_GRAVITY * R);
    theta_max_global = std::min(theta_max_global, M_PI);

    // Build spatial index if not already built or if size changed
    if (!spatial_index_built_ || spatial_index_.numFaces() != n) {
        std::cout << "Building spatial index (max query radius: "
                  << (theta_max_global * 180.0 / M_PI) << " deg)..." << std::endl;

        // Cell size should be smaller than typical query radius for efficiency
        double cell_size_deg = std::min(10.0, theta_max_global * 180.0 / M_PI / 3.0);
        cell_size_deg = std::max(2.0, cell_size_deg);  // But not too small

        spatial_index_.build(face_centers, cell_size_deg, theta_max_global * 1.1);
        spatial_index_query_radius_ = theta_max_global * 1.1;
        spatial_index_built_ = true;
        spatial_index_.printStatistics();
    }

    // Mean angular size for shell width
    double mean_theta = 0.0;
    for (double A : face_areas) mean_theta += std::sqrt(std::max(A, 0.0) / M_PI) / R;
    mean_theta /= std::max<size_t>(1, face_areas.size());

    // Shell factor
    double shell_factor = 2.5;
    if (const char* envp = std::getenv("EXO_SHELL_WIDTH_FACTOR")) {
        double v = std::atof(envp);
        if (v > 0.1 && v < 10.0) shell_factor = v;
    }
    const double dtheta_base = std::max(1e-6, shell_factor * mean_theta);
    const int max_shells = static_cast<int>(std::ceil(theta_max_global / dtheta_base)) + 2;

    bool disable_ring_corr = false;
    if (const char* envp = std::getenv("EXO_DISABLE_RING_CORR")) {
        disable_ring_corr = (std::atoi(envp) != 0);
    }

    // Kernel probability density function
    auto p_density = [&](double d, double T) {
        if (d <= 0.0 || T <= 0.0) return 0.0;
        double sigma2 = constants::BOLTZMANN_K * T / constants::WATER_MOLECULAR_MASS;
        double xi = constants::EUROPA_GRAVITY * d / (2.0 * sigma2);
        if (xi > 50.0) return 0.0;
        double sigma4 = sigma2 * sigma2;
        return (constants::EUROPA_GRAVITY * constants::EUROPA_GRAVITY * d / (4.0 * sigma4)) * besselK0(xi);
    };

    // Compute theta cutoff for a given temperature
    auto compute_theta_cutoff = [&](double T) {
        double sigma2 = constants::BOLTZMANN_K * T / constants::WATER_MOLECULAR_MASS;
        return 2.0 * sigma2 * xi_cutoff / (constants::EUROPA_GRAVITY * R);
    };

    // Self-cap integral using 6-point Gauss-Legendre
    auto selfCap = [&](double T, double A) {
        if (T <= 0.0 || A <= 0.0) return 0.0;
        constexpr int NG = 6;
        static const double xG[NG] = {-0.9324695142031520, -0.6612093864662645, -0.2386191860831969,
                                       0.2386191860831969,  0.6612093864662645,  0.9324695142031520};
        static const double wG[NG] = {0.1713244923791704, 0.3607615730481386, 0.4679139345726910,
                                      0.4679139345726910, 0.3607615730481386, 0.1713244923791704};
        double theta_cap = std::sqrt(std::max(A, 0.0) / (M_PI * R * R));
        theta_cap = std::min(theta_cap, 0.25);
        double a = 0, b = theta_cap;
        double integ = 0;
        for (int k = 0; k < NG; ++k) {
            double t = 0.5 * ((b - a) * xG[k] + (b + a));
            double d = R * t;
            double pd = p_density(d, T);
            double jac = 0.5 * (b - a);
            integ += wG[k] * pd * (2.0 * M_PI * R * R * std::sin(t)) * jac;
        }
        return integ;
    };

    std::cout << "Indexed matrix build: xi_cutoff=" << xi_cutoff
              << ", theta_max=" << (theta_max_global * 180.0 / M_PI) << " deg" << std::endl;

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(n * 64);

    // Statistics
    size_t total_queries = 0;
    size_t total_neighbors_found = 0;
    size_t total_pairs_evaluated = 0;

    #pragma omp parallel reduction(+:total_queries,total_neighbors_found,total_pairs_evaluated)
    {
        std::vector<Eigen::Triplet<double>> local;
        local.reserve(8192);

        // Thread-local neighbor results buffer
        std::vector<std::pair<size_t, float>> neighbors;
        neighbors.reserve(1024);

        std::vector<double> shell_disc_area(max_shells);
        std::vector<double> shell_cont_area(max_shells);
        std::vector<std::vector<std::pair<int, double>>> shell_entries(max_shells);

        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < n; ++i) {
            double T_i = temperatures[i];
            if (T_i <= 0.0) {
                local.emplace_back((int)i, (int)i, 1.0);
                continue;
            }

            // Compute per-source cutoff
            double theta_cutoff_i = compute_theta_cutoff(T_i);
            theta_cutoff_i = std::min(theta_cutoff_i, M_PI);
            float cos_cutoff_i = static_cast<float>(std::cos(theta_cutoff_i));

            // Escape fraction
            double vth = std::sqrt(2.0 * constants::BOLTZMANN_K * T_i / constants::WATER_MOLECULAR_MASS);
            double ratio = constants::EUROPA_ESCAPE_VELOCITY / vth;
            double escape_frac = 0.0;
            if (ratio < 5.0) {
                escape_frac = std::exp(-ratio * ratio) * (1.0 + ratio * ratio) / (2.0 * ratio * ratio);
            }

            double self_mass = selfCap(T_i, face_areas[i]);

            // Query spatial index for neighbors within cutoff
            total_queries++;
            spatial_index_.queryNeighborsByCosTheta(i, cos_cutoff_i, neighbors);
            total_neighbors_found += neighbors.size();

            // Reset shells
            std::fill(shell_disc_area.begin(), shell_disc_area.end(), 0.0);
            for (int s = 0; s < max_shells; ++s) shell_entries[s].clear();
            int used_shells = 0;

            // Process neighbors (already filtered by spatial index)
            for (const auto& [j, cos_theta] : neighbors) {
                total_pairs_evaluated++;

                // Compute theta from cos_theta
                double theta = std::acos(static_cast<double>(std::max(-1.0f, std::min(1.0f, cos_theta))));

                int s = static_cast<int>(std::floor(theta / dtheta_base));
                if (s >= max_shells) s = max_shells - 1;
                shell_disc_area[s] += face_areas[j];
                shell_entries[s].emplace_back(static_cast<int>(j), R * theta);
                if (s + 1 > used_shells) used_shells = s + 1;
            }

            // Compute continuous shell areas
            for (int s = 0; s < used_shells; ++s) {
                double theta_inner = s * dtheta_base;
                double theta_outer = (s + 1) * dtheta_base;
                if (theta_outer > theta_cutoff_i) theta_outer = theta_cutoff_i;
                if (theta_outer > theta_inner) {
                    shell_cont_area[s] =
                        2.0 * M_PI * R * R * (std::cos(theta_inner) - std::cos(theta_outer));
                } else {
                    shell_cont_area[s] = 0.0;
                }
            }

            double sum_off_raw = 0.0;
            std::vector<std::pair<int, double>> buf;
            buf.reserve(used_shells * 8);

            for (int s = 0; s < used_shells; ++s) {
                if (shell_entries[s].empty()) continue;
                double correction = 1.0;
                if (!disable_ring_corr && shell_disc_area[s] > 0.0 && shell_cont_area[s] > 0.0) {
                    correction = shell_cont_area[s] / shell_disc_area[s];
                }
                for (auto& rec : shell_entries[s]) {
                    int jidx = rec.first;
                    double d = rec.second;
                    double pd = p_density(d, T_i);
                    if (pd <= 0.0) continue;
                    double w_raw = pd * face_areas[jidx] * correction;
                    if (w_raw < threshold) continue;
                    buf.emplace_back(jidx, w_raw);
                    sum_off_raw += w_raw;
                }
            }

            double total_raw = self_mass + sum_off_raw;
            double target = 1.0 - escape_frac;
            if (total_raw <= 0.0) {
                local.emplace_back((int)i, (int)i, target);
                continue;
            }
            double scale = target / total_raw;
            local.emplace_back((int)i, (int)i, self_mass * scale);
            for (auto& pr : buf) {
                local.emplace_back(pr.first, (int)i, pr.second * scale);
            }
        }

        #pragma omp critical
        {
            triplets.insert(triplets.end(), local.begin(), local.end());
        }
    }

    Eigen::SparseMatrix<double> M(n, n);
    M.setFromTriplets(triplets.begin(), triplets.end());
    M.makeCompressed();

    auto build_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(build_end - build_start);

    // Report statistics
    double avg_neighbors = (total_queries > 0) ?
        static_cast<double>(total_neighbors_found) / static_cast<double>(total_queries) : 0.0;

    std::cout << "Transport matrix (indexed) built: " << n << " faces, "
              << M.nonZeros() << " non-zeros, " << duration.count() << " ms" << std::endl;
    std::cout << "  Index queries: " << total_queries
              << ", avg neighbors: " << std::fixed << std::setprecision(1) << avg_neighbors
              << ", pairs evaluated: " << total_pairs_evaluated << std::endl;

    verifyMassConservation(M);
    return M;
}

// NEW: Smooth kernel transport matrix - eliminates shell binning artifacts
// This method evaluates the kernel directly for each face pair without any
// discrete angular binning, avoiding the ring-density correction artifacts
// that occur with shell-based methods.
//
// IMPORTANT: This version uses AREA-NORMALIZED weighting to eliminate
// pentagon vertex artifacts. The key insight is that for equilibrium with
// uniform emission flux, we need:
//   Σ_i [M[j,i] * A_i] / A_j = 1  for all receiving faces j
//
// We achieve this by using normalized areas in the kernel integration:
//   w_ji = p(d_ji, T_i) * (A_j / A_mean)
// where A_mean is the mean face area. This ensures smaller faces don't
// receive systematically less mass per unit area.
Eigen::SparseMatrix<double> BallisticTransport::buildTransportMatrixSmooth(
    const std::vector<std::array<double, 3>>& face_centers,
    const std::vector<double>& face_areas,
    const std::vector<double>& temperatures,
    double threshold) const
{
    size_t n = face_centers.size();
    const double R = body_radius_;

    auto build_start = std::chrono::high_resolution_clock::now();

    // Compute mean face area for normalization
    double total_area = 0.0;
    for (size_t i = 0; i < n; ++i) {
        total_area += face_areas[i];
    }
    double mean_area = total_area / n;

    // Compute kernel cutoff parameters
    double xi_cutoff = 15.0;  // Default: captures 99.99% of transport
    if (const char* envp = std::getenv("EXO_XI_CUTOFF")) {
        double v = std::atof(envp);
        if (v > 1.0 && v < 100.0) xi_cutoff = v;
    }

    // Compute max theta for spatial index query radius
    double T_max = *std::max_element(temperatures.begin(), temperatures.end());
    T_max = std::max(T_max, 10.0);
    double sigma2_max = constants::BOLTZMANN_K * T_max / constants::WATER_MOLECULAR_MASS;
    double theta_max_global = 2.0 * sigma2_max * xi_cutoff / (constants::EUROPA_GRAVITY * R);
    theta_max_global = std::min(theta_max_global, M_PI);

    // Build spatial index if not already built or if size changed
    if (!spatial_index_built_ || spatial_index_.numFaces() != n) {
        std::cout << "Building spatial index for smooth kernel (max query radius: "
                  << (theta_max_global * 180.0 / M_PI) << " deg)..." << std::endl;

        double cell_size_deg = std::min(10.0, theta_max_global * 180.0 / M_PI / 3.0);
        cell_size_deg = std::max(2.0, cell_size_deg);

        spatial_index_.build(face_centers, cell_size_deg, theta_max_global * 1.1);
        spatial_index_query_radius_ = theta_max_global * 1.1;
        spatial_index_built_ = true;
    }

    // Kernel probability density function - direct evaluation, no binning
    // p(d, T) = (g²d / 4σ⁴) * K₀(gd / 2σ²)
    // where σ² = kT/m
    auto p_density = [&](double d, double T) -> double {
        if (d <= 0.0 || T <= 0.0) return 0.0;
        double sigma2 = constants::BOLTZMANN_K * T / constants::WATER_MOLECULAR_MASS;
        double xi = constants::EUROPA_GRAVITY * d / (2.0 * sigma2);
        if (xi > 50.0) return 0.0;
        double sigma4 = sigma2 * sigma2;
        return (constants::EUROPA_GRAVITY * constants::EUROPA_GRAVITY * d / (4.0 * sigma4)) * besselK0(xi);
    };

    // Compute theta cutoff for a given temperature
    auto compute_theta_cutoff = [&](double T) -> double {
        double sigma2 = constants::BOLTZMANN_K * T / constants::WATER_MOLECULAR_MASS;
        return 2.0 * sigma2 * xi_cutoff / (constants::EUROPA_GRAVITY * R);
    };

    // Self-deposition: integrate kernel over the source face itself
    // Using 6-point Gauss-Legendre quadrature over a spherical cap
    // NOTE: Use mean_area for self-deposition to maintain consistency
    auto selfCap = [&](double T, double A) -> double {
        if (T <= 0.0 || A <= 0.0) return 0.0;
        constexpr int NG = 6;
        static const double xG[NG] = {-0.9324695142031520, -0.6612093864662645, -0.2386191860831969,
                                       0.2386191860831969,  0.6612093864662645,  0.9324695142031520};
        static const double wG[NG] = {0.1713244923791704, 0.3607615730481386, 0.4679139345726910,
                                      0.4679139345726910, 0.3607615730481386, 0.1713244923791704};
        // Approximate face as spherical cap with equivalent area
        double theta_cap = std::sqrt(std::max(A, 0.0) / (M_PI * R * R));
        theta_cap = std::min(theta_cap, 0.25);
        double a = 0, b = theta_cap;
        double integ = 0;
        for (int k = 0; k < NG; ++k) {
            double t = 0.5 * ((b - a) * xG[k] + (b + a));
            double d = R * t;
            double pd = p_density(d, T);
            double jac = 0.5 * (b - a);
            // Differential area element on sphere: 2πR² sin(θ) dθ
            integ += wG[k] * pd * (2.0 * M_PI * R * R * std::sin(t)) * jac;
        }
        return integ;
    };

    std::cout << "Smooth kernel build: xi_cutoff=" << xi_cutoff
              << ", theta_max=" << (theta_max_global * 180.0 / M_PI) << " deg"
              << ", AREA-NORMALIZED (mean=" << mean_area << " m²)" << std::endl;

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(n * 64);

    // Statistics
    size_t total_queries = 0;
    size_t total_neighbors_found = 0;

    #pragma omp parallel reduction(+:total_queries,total_neighbors_found)
    {
        std::vector<Eigen::Triplet<double>> local;
        local.reserve(8192);

        // Thread-local neighbor results buffer
        std::vector<std::pair<size_t, float>> neighbors;
        neighbors.reserve(1024);

        // Buffer for raw weights before normalization
        std::vector<std::pair<int, double>> raw_weights;
        raw_weights.reserve(1024);

        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < n; ++i) {
            double T_i = temperatures[i];
            if (T_i <= 0.0) {
                local.emplace_back((int)i, (int)i, 1.0);
                continue;
            }

            // Compute per-source cutoff angle
            double theta_cutoff_i = compute_theta_cutoff(T_i);
            theta_cutoff_i = std::min(theta_cutoff_i, M_PI);
            float cos_cutoff_i = static_cast<float>(std::cos(theta_cutoff_i));

            // Escape fraction (Jeans escape)
            double vth = std::sqrt(2.0 * constants::BOLTZMANN_K * T_i / constants::WATER_MOLECULAR_MASS);
            double ratio = constants::EUROPA_ESCAPE_VELOCITY / vth;
            double escape_frac = 0.0;
            if (ratio < 5.0) {
                escape_frac = std::exp(-ratio * ratio) * (1.0 + ratio * ratio) / (2.0 * ratio * ratio);
            }

            // Self-deposition contribution (use actual area for self-cap calculation)
            double self_weight = selfCap(T_i, face_areas[i]);

            // Query spatial index for neighbors within cutoff
            total_queries++;
            spatial_index_.queryNeighborsByCosTheta(i, cos_cutoff_i, neighbors);
            total_neighbors_found += neighbors.size();

            // Compute raw transport weights for each neighbor
            // AREA-NORMALIZED: w_ji = p(d_ji, T_i) * (A_j / A_mean) * A_mean = p * A_j
            // But we normalize each column, so we can just use p * A_j and the
            // normalization will cancel out the A_mean factors.
            //
            // The key is that all faces use the same weighting scheme, so
            // pentagon-adjacent faces (with smaller A_j) get proportionally
            // smaller weights, but the column normalization ensures mass conservation.
            raw_weights.clear();
            double sum_weights = self_weight;

            for (const auto& [j, cos_theta] : neighbors) {
                // Compute great circle distance from cos(theta)
                double theta = std::acos(static_cast<double>(std::max(-1.0f, std::min(1.0f, cos_theta))));
                double d = R * theta;

                // Direct kernel evaluation - the smooth kernel approach
                double pd = p_density(d, T_i);
                if (pd <= 0.0) continue;

                // Weight = kernel value * face area (proper spherical integration)
                double w = pd * face_areas[j];
                if (w < threshold) continue;

                raw_weights.emplace_back(static_cast<int>(j), w);
                sum_weights += w;
            }

            // Normalize to ensure mass conservation
            // Total outgoing = (1 - escape_fraction)
            double target = 1.0 - escape_frac;
            if (sum_weights <= 0.0) {
                local.emplace_back((int)i, (int)i, target);
                continue;
            }

            double scale = target / sum_weights;

            // Self-deposition (diagonal entry)
            local.emplace_back((int)i, (int)i, self_weight * scale);

            // Off-diagonal entries (transport to other faces)
            for (const auto& [jidx, w] : raw_weights) {
                local.emplace_back(jidx, (int)i, w * scale);
            }
        }

        #pragma omp critical
        {
            triplets.insert(triplets.end(), local.begin(), local.end());
        }
    }

    Eigen::SparseMatrix<double> M(n, n);
    M.setFromTriplets(triplets.begin(), triplets.end());
    M.makeCompressed();

    // === AREA BIAS CORRECTION ===
    // The issue is that row sums (incoming transport) are biased toward faces
    // with larger areas. For equilibrium with uniform emission flux, we need:
    //   Σ_i [M[j,i] * A_i] = A_j  for all j
    //
    // Currently we have M[j,i] = (p_ji * A_j) / norm_i, which gives
    //   Σ_i [M[j,i] * A_i] = A_j * Σ_i [p_ji * A_i / norm_i]
    //
    // The bias comes from Σ_i [p_ji * A_i / norm_i] ≠ 1.
    //
    // We use an area-weighted Sinkhorn iteration that correctly targets:
    // - Column sums = 1 - escape_fraction (mass conservation)
    // - Area-weighted row sums: Σ_i[M[j,i] * A_i] = A_j (equilibrium)

    // Compute target column sums (mass conservation)
    std::vector<double> target_col_sums(n);
    for (size_t i = 0; i < n; ++i) {
        double vth = std::sqrt(2.0 * constants::BOLTZMANN_K * temperatures[i] / constants::WATER_MOLECULAR_MASS);
        double ratio = constants::EUROPA_ESCAPE_VELOCITY / vth;
        double escape_frac = 0.0;
        if (temperatures[i] > 0 && ratio < 5.0) {
            escape_frac = std::exp(-ratio * ratio) * (1.0 + ratio * ratio) / (2.0 * ratio * ratio);
        }
        target_col_sums[i] = 1.0 - escape_frac;
    }

    // Compute initial area-weighted row sums
    Eigen::VectorXd area_weighted_row_sum(n);
    area_weighted_row_sum.setZero();
    for (int k = 0; k < M.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(M, k); it; ++it) {
            area_weighted_row_sum(it.row()) += it.value() * face_areas[it.col()];
        }
    }

    // Compute bias factor for each face
    double max_bias = 0.0, min_bias = 1e10;
    for (size_t j = 0; j < n; ++j) {
        double bias = area_weighted_row_sum(j) / face_areas[j];
        max_bias = std::max(max_bias, bias);
        min_bias = std::min(min_bias, bias);
    }

    double bias_range = (max_bias - min_bias);

    // Use 2% threshold - small biases don't significantly affect results
    if (skip_sinkhorn_) {
        std::cout << "  Area bias range=" << std::fixed << std::setprecision(4)
                  << (bias_range * 100.0) << "% (Sinkhorn SKIPPED)" << std::endl;
    } else if (bias_range > 0.02) {
        std::cout << "  Area bias detected: range=" << std::fixed << std::setprecision(4)
                  << (bias_range * 100.0) << "%, applying area-weighted Sinkhorn..." << std::endl;

        // Area-weighted Sinkhorn iteration using SINGLE PRECISION for speed
        // Row target: Σ_i[M[j,i] * A_i] = A_j
        // Column target: Σ_j[M[j,i]] = 1 - escape_i

        const int max_sinkhorn_iters = 50;   // Reduced from 100
        const float sinkhorn_tol = 1e-4f;   // Relaxed tolerance for float
        const float bias_target = 0.005f;   // Early exit when bias < 0.5%

        // Use single precision for internal Sinkhorn computations
        std::vector<float> row_scale_f(n, 1.0f);
        std::vector<float> col_scale_f(n, 1.0f);
        std::vector<float> col_area_scale_f(n);
        std::vector<float> areas_f(n);
        std::vector<float> target_cols_f(n);

        // Convert to float
        for (size_t i = 0; i < n; ++i) {
            areas_f[i] = static_cast<float>(face_areas[i]);
            target_cols_f[i] = static_cast<float>(target_col_sums[i]);
            col_area_scale_f[i] = col_scale_f[i] * areas_f[i];
        }

        // Warm-start from cached scaling factors if available
        if (sinkhorn_cache_valid_ &&
            sinkhorn_row_scale_cache_.size() == n &&
            sinkhorn_col_scale_cache_.size() == n) {
            for (size_t i = 0; i < n; ++i) {
                row_scale_f[i] = static_cast<float>(sinkhorn_row_scale_cache_(i));
                col_scale_f[i] = static_cast<float>(sinkhorn_col_scale_cache_(i));
                col_area_scale_f[i] = col_scale_f[i] * areas_f[i];
            }
        }

        float* row_scale = row_scale_f.data();
        float* col_scale = col_scale_f.data();
        const float* areas = areas_f.data();
        const float* target_cols = target_cols_f.data();

        int converged_iter = max_sinkhorn_iters;

        // Pre-allocate sum arrays
        std::vector<float> current_aw_row_sums(n);
        std::vector<float> current_col_sums(n);

        for (int iter = 0; iter < max_sinkhorn_iters; ++iter) {
            // Step 1: Scale rows for target area-weighted row sums
            std::fill(current_aw_row_sums.begin(), current_aw_row_sums.end(), 0.0f);

            #pragma omp parallel
            {
                std::vector<float> local_row_sums(n, 0.0f);

                #pragma omp for nowait schedule(dynamic, 128)
                for (int k = 0; k < M.outerSize(); ++k) {
                    const float cas_k = col_area_scale_f[k];
                    for (Eigen::SparseMatrix<double>::InnerIterator it(M, k); it; ++it) {
                        local_row_sums[it.row()] += static_cast<float>(it.value()) * row_scale[it.row()] * cas_k;
                    }
                }

                #pragma omp critical
                {
                    for (size_t i = 0; i < n; ++i) {
                        current_aw_row_sums[i] += local_row_sums[i];
                    }
                }
            }

            // Update row scales
            float row_err = 0.0f;
            #pragma omp parallel for reduction(max:row_err)
            for (size_t j = 0; j < n; ++j) {
                if (current_aw_row_sums[j] > 1e-10f) {
                    float new_scale = areas[j] / current_aw_row_sums[j];
                    row_err = std::max(row_err, std::abs(new_scale - 1.0f));
                    row_scale[j] *= new_scale;
                }
            }

            // Step 2: Scale columns for target column sums
            std::fill(current_col_sums.begin(), current_col_sums.end(), 0.0f);

            #pragma omp parallel
            {
                std::vector<float> local_col_sums(n, 0.0f);

                #pragma omp for nowait schedule(dynamic, 128)
                for (int k = 0; k < M.outerSize(); ++k) {
                    float col_sum = 0.0f;
                    const float cs_k = col_scale[k];
                    for (Eigen::SparseMatrix<double>::InnerIterator it(M, k); it; ++it) {
                        col_sum += static_cast<float>(it.value()) * row_scale[it.row()] * cs_k;
                    }
                    local_col_sums[k] = col_sum;
                }

                #pragma omp critical
                {
                    for (size_t i = 0; i < n; ++i) {
                        current_col_sums[i] += local_col_sums[i];
                    }
                }
            }

            // Update col scales
            float col_err = 0.0f;
            #pragma omp parallel for reduction(max:col_err)
            for (size_t i = 0; i < n; ++i) {
                if (current_col_sums[i] > 1e-10f) {
                    float new_scale = target_cols[i] / current_col_sums[i];
                    col_err = std::max(col_err, std::abs(new_scale - 1.0f));
                    col_scale[i] *= new_scale;
                    col_area_scale_f[i] = col_scale[i] * areas[i];
                }
            }

            // Early termination check every 5 iterations based on actual bias
            if ((iter + 1) % 5 == 0 || (row_err < sinkhorn_tol && col_err < sinkhorn_tol)) {
                // Compute actual bias range
                float max_b = 0.0f, min_b = 1e10f;
                #pragma omp parallel for reduction(max:max_b) reduction(min:min_b)
                for (size_t j = 0; j < n; ++j) {
                    float bias = current_aw_row_sums[j] / areas[j];
                    max_b = std::max(max_b, bias);
                    min_b = std::min(min_b, bias);
                }
                float current_bias = max_b - min_b;

                if (current_bias < bias_target || (row_err < sinkhorn_tol && col_err < sinkhorn_tol)) {
                    converged_iter = iter + 1;
                    break;
                }
            }
        }

        // Cache scaling factors (convert back to double)
        sinkhorn_row_scale_cache_.resize(n);
        sinkhorn_col_scale_cache_.resize(n);
        for (size_t i = 0; i < n; ++i) {
            sinkhorn_row_scale_cache_(i) = static_cast<double>(row_scale[i]);
            sinkhorn_col_scale_cache_(i) = static_cast<double>(col_scale[i]);
        }
        sinkhorn_cache_valid_ = true;

        std::cout << "  Sinkhorn converged in " << converged_iter << " iterations" << std::endl;

        // Apply scaling to matrix (use double precision for final values)
        std::vector<Eigen::Triplet<double>> scaled_triplets;
        scaled_triplets.reserve(M.nonZeros());

        for (int k = 0; k < M.outerSize(); ++k) {
            const double cs_k = static_cast<double>(col_scale[k]);
            for (Eigen::SparseMatrix<double>::InnerIterator it(M, k); it; ++it) {
                double scaled_val = it.value() * static_cast<double>(row_scale[it.row()]) * cs_k;
                if (scaled_val > 1e-15) {
                    scaled_triplets.emplace_back(it.row(), it.col(), scaled_val);
                }
            }
        }

        M.setZero();
        M.setFromTriplets(scaled_triplets.begin(), scaled_triplets.end());
        M.makeCompressed();
    }

    auto build_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(build_end - build_start);

    // Report statistics
    double avg_neighbors = (total_queries > 0) ?
        static_cast<double>(total_neighbors_found) / static_cast<double>(total_queries) : 0.0;

    std::cout << "Transport matrix (smooth) built: " << n << " faces, "
              << M.nonZeros() << " non-zeros, " << duration.count() << " ms" << std::endl;
    std::cout << "  Index queries: " << total_queries
              << ", avg neighbors: " << std::fixed << std::setprecision(1) << avg_neighbors
              << std::endl;

    verifyMassConservation(M);
    return M;
}

// Matrix-free transport product: computes dep = M * emission without forming M.
// Two-pass source-based iteration with thread-local accumulation.
// Memory: O(N + N*threads) ≈ 90 MB for 1.3M faces, vs ~17 GB for full matrix.
void BallisticTransport::computeTransportProduct(
    const std::vector<std::array<double, 3>>& face_centers,
    const std::vector<double>& face_areas,
    const std::vector<double>& temperatures,
    const std::vector<double>& emission,
    std::vector<double>& deposition,
    double threshold) const
{
    size_t n = face_centers.size();
    const double R = body_radius_;

    auto start = std::chrono::high_resolution_clock::now();

    // Kernel cutoff: K0(15) ≈ 6e-8, captures >99.9999% of kernel weight.
    // Larger cutoff avoids visible sharp boundary at kernel edge.
    // At T=135K on Europa: theta_max ≈ 52° (vs 28° at xi=8).
    // Override with EXO_XI_CUTOFF env var.
    double xi_cutoff = 15.0;
    if (const char* envp = std::getenv("EXO_XI_CUTOFF")) {
        double v = std::atof(envp);
        if (v > 1.0 && v < 100.0) xi_cutoff = v;
    }

    double T_max = *std::max_element(temperatures.begin(), temperatures.end());
    T_max = std::max(T_max, 10.0);
    double sigma2_max = constants::BOLTZMANN_K * T_max / constants::WATER_MOLECULAR_MASS;
    double theta_max_global = 2.0 * sigma2_max * xi_cutoff / (constants::EUROPA_GRAVITY * R);
    theta_max_global = std::min(theta_max_global, M_PI);

    // Build spatial index if needed (persists across calls via mutable member)
    if (!spatial_index_built_ || spatial_index_.numFaces() != n) {
        double cell_size_deg = std::min(10.0, theta_max_global * 180.0 / M_PI / 3.0);
        cell_size_deg = std::max(2.0, cell_size_deg);
        spatial_index_.build(face_centers, cell_size_deg, theta_max_global * 1.1);
        spatial_index_query_radius_ = theta_max_global * 1.1;
        spatial_index_built_ = true;
    }

    // Kernel probability density with precomputed sigma2 for the source temperature.
    // Avoids redundant kT/m computation for each neighbor pair.
    auto p_density_fast = [&](double d, double sigma2, double sigma4) -> double {
        if (d <= 0.0) return 0.0;
        double xi = constants::EUROPA_GRAVITY * d / (2.0 * sigma2);
        if (xi > 50.0) return 0.0;
        return (constants::EUROPA_GRAVITY * constants::EUROPA_GRAVITY * d / (4.0 * sigma4)) * besselK0(xi);
    };

    // Fallback for selfCap which needs T-parameterized calls
    auto p_density = [&](double d, double T) -> double {
        if (d <= 0.0 || T <= 0.0) return 0.0;
        double sigma2 = constants::BOLTZMANN_K * T / constants::WATER_MOLECULAR_MASS;
        double xi = constants::EUROPA_GRAVITY * d / (2.0 * sigma2);
        if (xi > 50.0) return 0.0;
        double sigma4 = sigma2 * sigma2;
        return (constants::EUROPA_GRAVITY * constants::EUROPA_GRAVITY * d / (4.0 * sigma4)) * besselK0(xi);
    };

    // Self-deposition via Gauss-Legendre quadrature over spherical cap
    auto selfCap = [&](double T, double A) -> double {
        if (T <= 0.0 || A <= 0.0) return 0.0;
        constexpr int NG = 6;
        static const double xG[NG] = {-0.9324695142031520, -0.6612093864662645, -0.2386191860831969,
                                       0.2386191860831969,  0.6612093864662645,  0.9324695142031520};
        static const double wG[NG] = {0.1713244923791704, 0.3607615730481386, 0.4679139345726910,
                                      0.4679139345726910, 0.3607615730481386, 0.1713244923791704};
        double theta_cap = std::sqrt(std::max(A, 0.0) / (M_PI * R * R));
        theta_cap = std::min(theta_cap, 0.25);
        double a = 0, b = theta_cap;
        double integ = 0;
        for (int k = 0; k < NG; ++k) {
            double t = 0.5 * ((b - a) * xG[k] + (b + a));
            double d = R * t;
            double pd = p_density(d, T);
            double jac = 0.5 * (b - a);
            integ += wG[k] * pd * (2.0 * M_PI * R * R * std::sin(t)) * jac;
        }
        return integ;
    };

    // === SINGLE-PASS transport scatter ===
    // For each source: evaluate kernel weights for all neighbors within
    // cutoff, normalize to conserve mass (minus escape), then scatter
    // emission to neighbors. Row correction (calibration-based) handles
    // per-face density variation (pentagon vs hexagon regions).

    int num_threads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }

    // Thread-local deposition + calibration arrays
    // Calibration: scatter uniform unit flux (face_areas[i]) alongside actual emission
    // to measure area bias for Sinkhorn-like row correction.
    std::vector<std::vector<double>> thread_dep(num_threads, std::vector<double>(n, 0.0));
    std::vector<std::vector<double>> thread_cal(num_threads, std::vector<double>(n, 0.0));

    size_t total_scatter_ops = 0;

    #pragma omp parallel reduction(+:total_scatter_ops)
    {
        int tid = omp_get_thread_num();
        auto& local_dep = thread_dep[tid];
        auto& local_cal = thread_cal[tid];

        std::vector<std::pair<size_t, float>> neighbors;
        neighbors.reserve(1024);

        // Thread-local cache for neighbor weights (reused per source)
        std::vector<std::pair<size_t, double>> cached_weights;
        cached_weights.reserve(65536);

        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < n; ++i) {
            double e_i = emission[i];
            double T_i = temperatures[i];

            // Skip faces with no physical temperature
            if (T_i <= 0.0) continue;

            // Precompute per-source thermal parameters
            double sigma2_i = constants::BOLTZMANN_K * T_i / constants::WATER_MOLECULAR_MASS;
            double sigma4_i = sigma2_i * sigma2_i;

            // Escape fraction (Jeans escape) — reuse sigma2 for thermal velocity
            double vth = std::sqrt(2.0 * sigma2_i);
            double ratio = constants::EUROPA_ESCAPE_VELOCITY / vth;
            double escape_frac = 0.0;
            if (ratio < 5.0) {
                escape_frac = std::exp(-ratio * ratio) * (1.0 + ratio * ratio) / (2.0 * ratio * ratio);
            }

            double self_weight = selfCap(T_i, face_areas[i]);

            // Per-source angular cutoff
            double theta_cutoff_i = std::min(2.0 * sigma2_i * xi_cutoff / (constants::EUROPA_GRAVITY * R), M_PI);
            float cos_cutoff_i = static_cast<float>(std::cos(theta_cutoff_i));

            spatial_index_.queryNeighborsByCosTheta(i, cos_cutoff_i, neighbors);

            // Phase 1: compute kernel weights and cache them
            cached_weights.clear();

            const double xi_taper_start = 0.8 * xi_cutoff;
            const double xi_taper_range = xi_cutoff - xi_taper_start;

            for (const auto& [j, cos_theta] : neighbors) {
                double theta = FastAcos::evaluate(static_cast<double>(std::max(-1.0f, std::min(1.0f, cos_theta))));
                double d = R * theta;
                double pd = p_density_fast(d, sigma2_i, sigma4_i);
                if (pd <= 0.0) continue;

                // Smooth cosine taper near kernel cutoff to avoid circular disk artifacts
                double xi = constants::EUROPA_GRAVITY * d / (2.0 * sigma2_i);
                double taper = 1.0;
                if (xi > xi_taper_start) {
                    taper = 0.5 * (1.0 + std::cos(M_PI * (xi - xi_taper_start) / xi_taper_range));
                }

                double w = pd * face_areas[j] * taper;
                if (w < threshold) continue;
                cached_weights.emplace_back(j, w);
            }

            // Sum weights for normalization. Per-source normalization ensures
            // each source scatters exactly (1 - escape_frac) total flux.
            // The row correction (calibration-based) handles per-face density
            // variation without introducing shell-boundary artifacts.
            double sum_weights = self_weight;
            for (const auto& [j_w, w] : cached_weights) {
                sum_weights += w;
            }

            // Normalization
            double target = 1.0 - escape_frac;
            double norm_i = (sum_weights > 0.0) ? target / sum_weights : 0.0;

            // Calibration: uniform unit flux from every face (emission = face_area)
            double scaled_a = face_areas[i] * norm_i;

            // Self-deposition (calibration always, actual only if emitting)
            local_cal[i] += self_weight * scaled_a;
            if (e_i > 0.0) {
                double scaled_e = e_i * norm_i;
                local_dep[i] += self_weight * scaled_e;
                total_scatter_ops++;

                // Phase 2: scatter actual emission
                for (const auto& [j, w] : cached_weights) {
                    local_dep[j] += w * scaled_e;
                    total_scatter_ops++;
                }
            }

            // Phase 2b: scatter calibration (always, for all faces with T > 0)
            for (const auto& [j, w] : cached_weights) {
                local_cal[j] += w * scaled_a;
            }
        }
    }

    // Merge pass 1 thread-local results
    std::vector<double> calibration(n, 0.0);
    deposition.assign(n, 0.0);
    for (int t = 0; t < num_threads; ++t) {
        const auto& td = thread_dep[t];
        const auto& tc = thread_cal[t];
        #pragma omp parallel for
        for (size_t j = 0; j < n; ++j) {
            deposition[j] += td[j];
            calibration[j] += tc[j];
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Pre-correction mass conservation
    double total_emission = 0.0, total_deposition = 0.0;
    #pragma omp parallel for reduction(+:total_emission,total_deposition)
    for (size_t i = 0; i < n; ++i) {
        total_emission += emission[i];
        total_deposition += deposition[i];
    }

    // === ROW-SUM BIAS CORRECTION ===
    // The calibration vector cal[j] measures total deposition at face j for
    // uniform emission. On a perfect mesh, cal[j] = area[j] * (1-escape).
    // Pentagon-adjacent faces have different local face density than hexagon
    // faces, causing systematic bias of ~20-30%. The correction factor
    // bias[j] = area[j] / cal[j] directly corrects this mesh discretization
    // error. This does NOT cancel the K0 kernel shape because cal[j] is a
    // sum over all ~300K+ source faces — no single source dominates it.
    bool disable_bias_correction = false;
    if (const char* envp = std::getenv("EXO_DISABLE_BIAS_CORR")) {
        disable_bias_correction = (std::atoi(envp) != 0);
    }

    // Compute per-face bias: expected (area) / actual (calibration) response
    std::vector<double> bias(n, 1.0);
    double max_bias = 0.0;
    #pragma omp parallel for reduction(max:max_bias)
    for (size_t j = 0; j < n; ++j) {
        if (calibration[j] > 1e-30) {
            bias[j] = face_areas[j] / calibration[j];
            max_bias = std::max(max_bias, std::abs(bias[j] - 1.0));
        }
    }

    double max_residual_bias = max_bias;

    if (!disable_bias_correction && max_bias > 0.01) {
        // Apply raw bias correction — exact for uniform emission,
        // first-order accurate for non-uniform emission
        #pragma omp parallel for
        for (size_t j = 0; j < n; ++j) {
            deposition[j] *= bias[j];
        }
        max_residual_bias = 0.0;  // Exact correction for calibration case
    }

    // === SPATIAL SMOOTHING (off by default) ===
    // Radius from spatialSmoothingRadiusRad(): 0 unless EXO_DEP_SMOOTH_DEG opts
    // in, and the same radius the sublimation/sputtering fields use, so
    // net_flux = dep - sub stays at one resolution.
    const double smooth_radius = spatialSmoothingRadiusRad();
    const double smooth_radius_deg = smooth_radius * 180.0 / M_PI;
    if (smooth_radius > 0.0) {
        if (smooth_radius > spatial_index_query_radius_) {
            throw std::invalid_argument("EXO_DEP_SMOOTH_DEG exceeds the spatial index query radius ("
                + std::to_string(spatial_index_query_radius_ * 180.0 / M_PI) + " deg)");
        }
        float cos_smooth = static_cast<float>(std::cos(smooth_radius));
        std::vector<double> dep_smooth(n);

        #pragma omp parallel
        {
            std::vector<std::pair<size_t, float>> smooth_neighbors;
            smooth_neighbors.reserve(512);

            #pragma omp for schedule(dynamic)
            for (size_t j = 0; j < n; ++j) {
                spatial_index_.queryNeighborsByCosTheta(j, cos_smooth, smooth_neighbors);

                double wsum = face_areas[j];
                double dep_sum = deposition[j] * face_areas[j];
                for (const auto& [k, cos_theta] : smooth_neighbors) {
                    dep_sum += deposition[k] * face_areas[k];
                    wsum += face_areas[k];
                }
                dep_smooth[j] = dep_sum / wsum;
            }
        }
        deposition = std::move(dep_smooth);
    }

    // Restore global mass conservation (after smoothing)
    double total_dep_corrected = 0.0;
    #pragma omp parallel for reduction(+:total_dep_corrected)
    for (size_t j = 0; j < n; ++j) {
        total_dep_corrected += deposition[j];
    }
    if (total_dep_corrected > 1e-30 && total_deposition > 1e-30) {
        double mass_fix = total_deposition / total_dep_corrected;
        #pragma omp parallel for
        for (size_t j = 0; j < n; ++j) {
            deposition[j] *= mass_fix;
        }
    }

    // Final conservation check
    double total_dep_final = 0.0;
    #pragma omp parallel for reduction(+:total_dep_final)
    for (size_t j = 0; j < n; ++j) {
        total_dep_final += deposition[j];
    }
    double conservation_ratio = (total_emission > 0.0) ? total_dep_final / total_emission : 1.0;

    auto end2 = std::chrono::high_resolution_clock::now();
    auto ms_total = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start);

    std::cout << "Matrix-free transport: " << n << " faces, "
              << total_scatter_ops << " scatter ops, " << ms_total.count() << " ms"
              << ", conservation=" << std::fixed << std::setprecision(4)
              << (conservation_ratio * 100.0) << "%"
              << ", area-bias=" << std::setprecision(1) << (max_bias * 100.0) << "%"
              << (disable_bias_correction ? " (uncorrected)" : " (row-corrected)")
              << ", smoothing="
              << (smooth_radius > 0.0 ? std::to_string(smooth_radius_deg) + " deg" : std::string("off"))
              << std::endl;
}

// ExosphereModel wrapper for matrix-free transport product
void ExosphereModel::computeTransportProduct(
    const std::vector<double>& temperatures,
    const std::vector<double>& emission,
    std::vector<double>& deposition)
{
    transport_->computeTransportProduct(
        face_centers_, face_areas_, temperatures, emission, deposition);
}

// Smooth a scalar field using the spatial index built by computeTransportProduct().
// Area-weighted averaging within smooth_radius_rad.
void BallisticTransport::smoothField(
    const std::vector<double>& face_areas,
    const std::vector<double>& input,
    std::vector<double>& output,
    double smooth_radius_rad) const
{
    size_t n = input.size();
    if (!spatial_index_built_ || n == 0 || smooth_radius_rad <= 0.0) {
        output = input;  // radius <= 0: smoothing off, exact no-op
        return;
    }
    if (smooth_radius_rad > spatial_index_query_radius_) {
        throw std::invalid_argument("smoothField radius exceeds the spatial index query radius ("
            + std::to_string(spatial_index_query_radius_ * 180.0 / M_PI) + " deg)");
    }

    float cos_smooth = static_cast<float>(std::cos(smooth_radius_rad));
    output.resize(n);

    #pragma omp parallel
    {
        std::vector<std::pair<size_t, float>> neighbors;
        neighbors.reserve(512);

        #pragma omp for schedule(dynamic)
        for (size_t j = 0; j < n; ++j) {
            spatial_index_.queryNeighborsByCosTheta(j, cos_smooth, neighbors);

            double wsum = face_areas[j];
            double vsum = input[j] * face_areas[j];
            for (const auto& [k, cos_theta] : neighbors) {
                vsum += input[k] * face_areas[k];
                wsum += face_areas[k];
            }
            output[j] = vsum / wsum;
        }
    }
}

// ExosphereModel wrapper for smoothField
void ExosphereModel::smoothField(
    const std::vector<double>& input,
    std::vector<double>& output,
    double smooth_radius_rad)
{
    if (transport_) {
        transport_->smoothField(face_areas_, input, output, smooth_radius_rad);
    } else {
        output = input;
    }
}

} // namespace exosphere
} // namespace ptm