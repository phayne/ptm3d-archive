#include "thermal_model.h"
#include "../maps/spatial_map.h"
#include "eigen_config.h"
#include <iostream>
#include <algorithm>
#include <cmath>

// CPU feature detection - protect against redefinition
#if defined(__AVX512F__) && !defined(AVX512_ENABLED)
#include <immintrin.h>
#define AVX512_ENABLED
#elif !defined(AVX2_ENABLED) && defined(__AVX2__)
#include <immintrin.h>
#define AVX2_ENABLED
#endif

// Constants for algorithm selection
#define THOMAS_MAX_SIZE 64  // Maximum size where Thomas algorithm is preferred
#define MIN_VECTORIZED_SIZE 10  // Minimum size for vectorized processing

namespace thermal {

// Thomas algorithm for tridiagonal systems
// a - lower diagonal (n elements)
// b - main diagonal (n elements)
// c - upper diagonal (n elements)
// d - right-hand side (n elements, will be overwritten with solution)
void solveTridiagonal(double* a, double* b, double* c, double* d, int n) {
    // Forward sweep
    for (int i = 1; i < n; i++) {
        double m = a[i] / b[i-1];
        b[i] -= m * c[i-1];
        d[i] -= m * d[i-1];
    }
    
    // Back substitution
    d[n-1] /= b[n-1];
    for (int i = n-2; i >= 0; i--) {
        d[i] = (d[i] - c[i] * d[i+1]) / b[i];
    }
    
    // Solution is now in d
}

void solveTridiagonalOptimized(double* a, double* b, double* c, double* d, int n) {
    // Small matrices - use scalar Thomas algorithm (current implementation)
    if (n <= THOMAS_MAX_SIZE) {
        solveTridiagonal(a, b, c, d, n);
        return;
    }
    
    // Large matrices - use parallel cyclic reduction or PCR
    // This algorithm has better parallelism for large systems
    #pragma omp parallel
    {
        // Phase 1: Forward elimination (can be partially parallelized)
        #pragma omp for nowait
        for (int i = 0; i < n; i++) {
            // Normalize equation i
            double invB = 1.0 / b[i];
            if (i > 0) a[i] *= invB;
            d[i] *= invB;
            b[i] = 1.0;
            
            // Forward elimination for odd-indexed rows
            if (i < n-1 && (i & 1)) {
                double factor = c[i];
                c[i] = 0.0;
                b[i+1] -= factor * a[i+1];
                d[i+1] -= factor * d[i];
            }
        }
        
        // Phase 2: Backward substitution
        #pragma omp for
        for (int i = n-1; i >= 0; i--) {
            if (i < n-1) {
                d[i] -= c[i] * d[i+1];
            }
        }
    }
}

// Essential helper functions
double R350(double chi) {
    return chi / std::pow(350.0, 3);
}

double skinDepth(double P, double kappa) {
    return std::sqrt(kappa * P / M_PI);
}

std::vector<double> spatialGrid(double zs, int m, int n, int b) {
    std::vector<double> z(1, 0.0);      // initialize depth array at zero
    double dz0 = zs / m;                // thickness of uppermost model layer
    
    // Diagnostic output
    std::cout << "Creating spatial grid with skin depth: " << zs << "m" << std::endl;
    std::cout << "Uppermost layer thickness: " << dz0 << "m" << std::endl;
    
    double zmax = zs * b;               // depth of deepest model layer
    double current_dz = dz0;
    
    while (z.back() < zmax) {
        z.push_back(z.back() + current_dz);
        current_dz *= (1.0 + 1.0 / n);  // geometric increase
    }
    
    std::cout << "Created grid with " << z.size() << " layers to depth " << z.back() << "m" << std::endl;
    return z;
}

std::vector<double> generateStableGrid(double skin_depth, int m, int n, int b, bool verbose = false) {
    std::vector<double> z(1, 0.0);      // initialize at zero
    
    // Set minimum layer thickness based on skin depth
    double min_dz = std::max(0.001, skin_depth / 100.0);  // sensible minimum
    
    // Calculate initial thickness with safety factor
    double dz0 = std::max(skin_depth / m, min_dz);
    
    // Only print if verbose is true
    if (verbose) {
        std::cout << "Creating spatial grid with:" << std::endl;
        std::cout << "  Skin depth: " << skin_depth << " m" << std::endl;
        std::cout << "  Uppermost layer thickness: " << dz0 << " m" << std::endl;
        std::cout << "  Minimum allowed thickness: " << min_dz << " m" << std::endl;
    }
    
    // Calculate maximum depth
    double zmax = skin_depth * b;
    double current_dz = dz0;
    
    // Build grid with thickness safety checks
    while (z.back() < zmax) {
        // Calculate next layer thickness with geometric increase
        double next_dz = current_dz * (1.0 + 1.0/n);
        
        // Enforce minimum thickness
        next_dz = std::max(next_dz, min_dz);
        
        // Add new layer
        z.push_back(z.back() + next_dz);
        current_dz = next_dz;
    }
    
    // Only print if verbose is true
    if (verbose) {
        std::cout << "Created grid with " << z.size() << " layers to depth " 
                  << z.back() << " m" << std::endl;
    }
    
    return z;
}

double T_radeq(const Planet& planet, double lat) {
    double solar_flux = planet.S * std::cos(lat);
    return std::pow((1.0 - planet.albedo) * solar_flux / (SIGMA_SB * planet.emissivity), 0.25);
}

double T_eq(const Planet& planet, double lat) {
    double solar_flux = planet.S * std::cos(lat);
    return std::pow((1.0 - planet.albedo) * solar_flux / (SIGMA_SB * planet.emissivity), 0.25) / std::sqrt(2.0);
}

// Planet class implementation
Planet::Planet(const std::string& name) {
    this->name = name;
    
    // Load properties from the database based on name
    ThermophysicalProperties props = PlanetaryData::getPropertiesByName(name);
    
    // Apply the properties
    albedo = props.albedo;
    emissivity = props.emissivity;
    day = props.day;
    year = props.year;
    rAU = props.rAU;
    S = props.S;
    Qb = props.Qb;
    ks = props.ks;
    kd = props.kd;
    rhos = props.rhos;
    rhod = props.rhod;
    H = props.H;
    cp0 = props.cp0;
    cpCoeff = props.cpCoeff;
    albedoCoef = props.albedoCoef;
}

Planet::Planet(PlanetaryBody body) {
    // Set name based on body type
    switch (body) {
        case PlanetaryBody::EUROPA: name = "Europa"; break;
        case PlanetaryBody::MOON: name = "Moon"; break;
        case PlanetaryBody::MERCURY: name = "Mercury"; break;
        default: name = "Custom"; break;
    }
    
    // Load properties from the database
    ThermophysicalProperties props = PlanetaryData::getProperties(body);
    
    // Apply the properties
    albedo = props.albedo;
    emissivity = props.emissivity;
    day = props.day;
    year = props.year;
    rAU = props.rAU;
    S = props.S;
    Qb = props.Qb;
    ks = props.ks;
    kd = props.kd;
    rhos = props.rhos;
    rhod = props.rhod;
    H = props.H;
    cp0 = props.cp0;
    cpCoeff = props.cpCoeff;
    albedoCoef = props.albedoCoef;
}

bool Planet::loadAlbedoMap(const std::string& filename, bool verbose) {
    if (verbose) {
        std::cout << "Loading albedo map from " << filename << std::endl;
    }
    
    try {
        albedo_map = std::make_shared<ptm::SpatialMap>();
        if (!albedo_map->loadFromFITS(filename)) {
            std::cerr << "Error loading albedo map: " << albedo_map->getLastError() << std::endl;
            albedo_map.reset();
            return false;
        }
        
        using_albedo_map = true;
        
        if (verbose) {
            std::cout << "Albedo map loaded, resolution: " 
                      << albedo_map->getWidth() << "x" << albedo_map->getHeight() 
                      << " pixels" << std::endl;
            std::cout << "Albedo range: " << albedo_map->getMin() 
                      << " to " << albedo_map->getMax() << std::endl;
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception loading albedo map: " << e.what() << std::endl;
        albedo_map.reset();
        using_albedo_map = false;
        return false;
    }
}

bool Planet::loadThermalInertiaMap(const std::string& filename, bool verbose) {
    if (verbose) {
        std::cout << "Loading thermal inertia map from " << filename << std::endl;
    }
    
    try {
        thermal_inertia_map = std::make_shared<ptm::SpatialMap>();
        if (!thermal_inertia_map->loadFromFITS(filename)) {
            std::cerr << "Error loading thermal inertia map: " 
                      << thermal_inertia_map->getLastError() << std::endl;
            thermal_inertia_map.reset();
            return false;
        }
        
        using_thermal_inertia_map = true;
        
        if (verbose) {
            std::cout << "Thermal inertia map loaded, resolution: " 
                      << thermal_inertia_map->getWidth() << "x" << thermal_inertia_map->getHeight() 
                      << " pixels" << std::endl;
            std::cout << "Thermal inertia range: " << thermal_inertia_map->getMin() 
                      << " to " << thermal_inertia_map->getMax() << std::endl;
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception loading thermal inertia map: " << e.what() << std::endl;
        thermal_inertia_map.reset();
        using_thermal_inertia_map = false;
        return false;
    }
}

double Planet::getAlbedo(double lon, double lat) const {
    if (using_albedo_map && albedo_map) {
        double map_value = albedo_map->getValue(lon, lat);
        
        // If we got a valid value from the map, use it
        if (map_value != albedo_map->getMissingValue()) {
            return map_value;
        }
    }
    
    // Fall back to the global albedo if no map or value not found
    return albedo;
}

double Planet::getThermalInertia(double lon, double lat) const {
    if (using_thermal_inertia_map && thermal_inertia_map) {
        double map_value = thermal_inertia_map->getValue(lon, lat);
        
        // If we got a valid value from the map, use it
        if (map_value != thermal_inertia_map->getMissingValue()) {
            return map_value;
        }
    }
    
    // Calculate default thermal inertia if no map or value not found
    // Thermal inertia = sqrt(k * rho * cp)
    double k = ks + 0.01 * std::pow(80.0, 3);  // Default conductivity at ~80K
    double rho_val = rhos;  // Default density
    double cp_val = cp0;    // Default heat capacity
    
    return std::sqrt(k * rho_val * cp_val);
}

void Planet::setProperty(const std::string& property_name, double value) {
    // Convert to lowercase for case-insensitive comparison
    std::string prop = property_name;
    for (auto& c : prop) c = std::tolower(c);
    
    // Set the appropriate property
    if (prop == "albedo") albedo = value;
    else if (prop == "emissivity") emissivity = value;
    else if (prop == "day") day = value;
    else if (prop == "year") year = value;
    else if (prop == "rau") rAU = value;
    else if (prop == "s") S = value;
    else if (prop == "qb") Qb = value;
    else if (prop == "ks") ks = value;
    else if (prop == "kd") kd = value;
    else if (prop == "rhos") rhos = value;
    else if (prop == "rhod") rhod = value;
    else if (prop == "h") H = value;
    else if (prop == "cp0") cp0 = value;
    else {
        std::cerr << "Warning: Unknown property '" << property_name << "'" << std::endl;
    }
}

void Planet::setVectorProperty(const std::string& property_name, const std::vector<double>& values) {
    // Convert to lowercase for case-insensitive comparison
    std::string prop = property_name;
    for (auto& c : prop) c = std::tolower(c);
    
    // Set the appropriate property
    if (prop == "cpcoeff") cpCoeff = values;
    else if (prop == "albedocoef") albedoCoef = values;
    else {
        std::cerr << "Warning: Unknown vector property '" << property_name << "'" << std::endl;
    }
}

// Profile class implementation
Profile::Profile(const Planet& planet, double lat, const Config& config, bool verbose) {
    this->planet = planet;
    this->config = &config;
    this->emissivity = planet.emissivity;
    this->chi = config.chi;
    this->R350_val = config.getR350();
    this->day = planet.day;
    
    // Calculate thermal parameters
    double kappa = planet.ks / (planet.rhos * planet.cp0);
    double skin_d = skinDepth(planet.day, kappa);
    
    // Use a minimum skin depth to prevent very small layers
    if (skin_d < 0.01)  // Minimum 1cm skin depth
        skin_d = 0.01;
    
    // Create spatial grid - pass the verbose flag
    z() = generateStableGrid(skin_d, config.m, config.n, config.b, verbose);
    nlayers = z().size();
    
    // Calculate layer thicknesses
    dz().resize(z().size() - 1);
    for (int i = 0; i < z().size() - 1; i++) {
        dz()[i] = z()[i+1] - z()[i];
        
        // Safety check for very thin layers
        if (dz()[i] < 0.001) dz()[i] = 0.001;
    }
    
    // Calculate grid coefficients for finite difference
    d3z().resize(dz().size() - 1);
    g1().resize(dz().size() - 1);
    g2().resize(dz().size() - 1);
    for (int i = 0; i < dz().size() - 1; i++) {
        d3z()[i] = dz()[i+1] * dz()[i] * (dz()[i+1] + dz()[i]);
        g1()[i] = 2.0 * dz()[i+1] / d3z()[i];
        g2()[i] = 2.0 * dz()[i] / d3z()[i];
    }
    
    // Initialize thermal properties arrays
    kc().resize(nlayers);
    rho().resize(nlayers);
    for (int i = 0; i < nlayers; i++) {
        double depth = z()[i];
        double depth_factor = std::exp(-depth / planet.H);
        
        // Depth-dependent properties
        kc()[i] = planet.kd + (planet.ks - planet.kd) * depth_factor;
        rho()[i] = planet.rhod + (planet.rhos - planet.rhod) * depth_factor;
    }
    
    // Initialize temperature profile
    init_T(planet, lat, verbose);
    
    // Initialize conductivity and heat capacity profiles
    update_k();
    update_cp();
    
    // Only show initialization messages if verbose
    if (verbose) {
        std::cout << "Initialized thermal profile with " << nlayers << " layers" << std::endl;
        std::cout << "Skin depth: " << skin_d << " m" << std::endl;
        std::cout << "Layer thicknesses: " << dz()[0] << " to " << dz()[dz().size()-1] << " m" << std::endl;
        std::cout << "Total depth: " << z()[z().size()-1] << " m" << std::endl;
    }
    
    // Set up aligned memory for better SIMD performance
    alignas(64) std::vector<double> aligned_T(T().size()); 
    std::copy(T().begin(), T().end(), aligned_T.begin());
    T() = std::move(aligned_T);
}

void Profile::init_T(const Planet& planet, double lat, bool verbose) {
    // Calculate radiative equilibrium temperature for peak insolation
    double T_max = std::pow((1.0 - planet.albedo) * planet.S / (SIGMA_SB * planet.emissivity), 0.25);
    
    // Only print if verbose is true
    if (verbose) {
        std::cout << "Maximum radiative equilibrium temperature: " << T_max << "K" << std::endl;
    }
    
    // Calculate average temperature (more realistic starting point)
    double T_avg = T_max / std::sqrt(2.0);
    
    // Calculate deep temperature based on internal heat flux
    double k_deep = planet.kd + R350_val * std::pow(80.0, 3);
    double gradient = planet.Qb / k_deep;
    double depth = z().back();
    double T_deep = T_avg + gradient * depth;
    
    // Only print if verbose
    if (verbose) {
        std::cout << "Initializing temperature profile:" << std::endl;
        std::cout << "  Surface average: " << T_avg << "K" << std::endl;
        std::cout << "  Deep temperature: " << T_deep << "K" << std::endl;
    }
    
    // Initialize with depth-dependent profile
    T().resize(nlayers);
    for (size_t i = 0; i < nlayers; i++) {
        // Exponential blend from surface to deep temperature
        double blend = std::exp(-z()[i] / (skinDepth(planet.day, planet.ks/(planet.rhos*planet.cp0))));
        T()[i] = T_deep + (T_avg - T_deep) * blend;
    }
}

void Profile::update_cp() {
    cp().resize(T().size());
    
    // For improved efficiency, use thread-level parallelism for large arrays
    int num_threads = 1;
#ifdef _OPENMP
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }
#endif

    if (num_threads > 1 && T().size() > 1000) {
        // Parallel implementation for large arrays
        #pragma omp parallel for
        for (size_t i = 0; i < T().size(); i++) {
            cp()[i] = planet.cp0;
            for (size_t j = 0; j < planet.cpCoeff.size(); j++) {
                cp()[i] += planet.cpCoeff[j] * std::pow(T()[i], j+1);
            }
        }
    } else {
        // Scalar implementation for smaller arrays or single thread
        for (size_t i = 0; i < T().size(); i++) {
            cp()[i] = planet.cp0;
            for (size_t j = 0; j < planet.cpCoeff.size(); j++) {
                cp()[i] += planet.cpCoeff[j] * std::pow(T()[i], j+1);
            }
        }
    }
}

// Fix the update_k() method to only calculate thermal conductivity
void Profile::update_k() {
    k().resize(T().size());
    
    // Check if running in multi-threaded mode - use simple implementation for better scaling
    int num_threads = 1;
#ifdef _OPENMP
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }
#endif

    if (num_threads > 1) {
        // Multi-threaded scalar implementation
        #pragma omp parallel for
        for (size_t i = 0; i < T().size(); i++) {
            k()[i] = kc()[i] + R350_val * std::pow(T()[i], 3);
        }
    }
    else {
#if defined(AVX512_ENABLED)
        // AVX-512 optimized implementation - process 8 elements at once
        const size_t vec_size = 8;
        const size_t vec_end = T().size() - (T().size() % vec_size);
        
        // Process blocks of 8 elements
        for (size_t i = 0; i < vec_end; i += vec_size) {
            // Load 8 temperature values using unaligned load for flexibility
            __m512d T_vec = _mm512_loadu_pd(&T()[i]);
            
            // Load 8 kc values 
            __m512d kc_vec = _mm512_loadu_pd(&kc()[i]);
            
            // Calculate T^2 and then T^3 (faster than pow)
            __m512d T_squared = _mm512_mul_pd(T_vec, T_vec);
            __m512d T_cubed = _mm512_mul_pd(T_squared, T_vec);
            
            // Calculate R350_val * T^3
            __m512d radiative_term = _mm512_mul_pd(_mm512_set1_pd(R350_val), T_cubed);
            
            // Calculate kc + R350_val * T^3
            __m512d result = _mm512_add_pd(kc_vec, radiative_term);
            
            // Store the result using unaligned store
            _mm512_storeu_pd(&k()[i], result);
        }
        
        // Handle remaining elements with scalar code
        for (size_t i = vec_end; i < T().size(); i++) {
            k()[i] = kc()[i] + R350_val * std::pow(T()[i], 3);
        }
#elif defined(AVX2_ENABLED)
        // AVX2 implementation for thermal conductivity
        const size_t vec_size = 4;
        const size_t vec_end = T().size() - (T().size() % vec_size);
        
        for (size_t i = 0; i < vec_end; i += vec_size) {
            // Load 4 temperature values
            __m256d T_vec = _mm256_loadu_pd(&T()[i]);
            
            // Load 4 kc values
            __m256d kc_vec = _mm256_loadu_pd(&kc()[i]);
            
            // Calculate T^2 and then T^3
            __m256d T_squared = _mm256_mul_pd(T_vec, T_vec);
            __m256d T_cubed = _mm256_mul_pd(T_squared, T_vec);
            
            // Calculate R350_val * T^3
            __m256d radiative_term = _mm256_mul_pd(_mm256_set1_pd(R350_val), T_cubed);
            
            // Calculate kc + R350_val * T^3
            __m256d result = _mm256_add_pd(kc_vec, radiative_term);
            
            // Store the result
            _mm256_storeu_pd(&k()[i], result);
        }
        
        // Handle remaining elements
        for (size_t i = vec_end; i < T().size(); i++) {
            k()[i] = kc()[i] + R350_val * std::pow(T()[i], 3);
        }
#else
        // Scalar implementation with manual optimization
        for (size_t i = 0; i < T().size(); i++) {
            // Use direct multiplication instead of std::pow for better performance
            double t = T()[i];
            k()[i] = kc()[i] + R350_val * (t * t * t);
        }
#endif
    }
}

void Profile::update_T(double dt, double Qs, double Qb) {
    // Apply boundary conditions first
    surfTemp(*this, Qs);
    if (Qb != 0) botTemp(*this, Qb);
    
    // Get number of interior points (exclude boundaries)
    const size_t n = grid.T.size() - 2;
    if (n < 1) return; // Nothing to do for very small profiles
    
    // Create aligned temperature cache for better memory access patterns
    alignas(64) static thread_local std::vector<double> T_old;
    if (T_old.size() < grid.T.size()) {
        T_old.resize(grid.T.size(), 0.0);
    }
    
    // Copy current temperatures to cache
    std::copy(grid.T.begin(), grid.T.end(), T_old.begin());
    
    // Detect if running in multi-threaded mode
    int num_threads = 1;
#ifdef _OPENMP
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }
#endif
    
    // Choose algorithm based on thread count and problem size
    if (num_threads > 1) {
        // Multi-threaded mode: use simple point-Jacobi iteration for better scalability
        #pragma omp parallel for
        for (size_t i = 1; i < grid.T.size() - 1; i++) {
            // Calculate thermal diffusivity
            double kappa = grid.k[i] / (grid.rho[i] * grid.cp[i]);
            double r = dt * kappa;
            
            // Calculate Crank-Nicolson coefficients
            double a_coef = -0.5 * r * grid.g1[i-1];
            double c_coef = -0.5 * r * grid.g2[i-1];
            double b_coef = 1.0 + 0.5 * r * (grid.g1[i-1] + grid.g2[i-1]);
            
            // Calculate right-hand side
            double rhs_val = T_old[i] + 0.5 * r * (grid.g1[i-1] * T_old[i-1] + 
                                    grid.g2[i-1] * T_old[i+1] - 
                                    (grid.g1[i-1] + grid.g2[i-1]) * T_old[i]);
            
            // Solve for new temperature
            grid.T[i] = (rhs_val - a_coef * grid.T[i-1] - c_coef * grid.T[i+1]) / b_coef;
        }
    }
    else if (n <= THOMAS_MAX_SIZE) {
        // For small systems, use direct Thomas algorithm with pre-allocated memory
        memory_pool.ensure_capacity(n);
        double* a = memory_pool.a();
        double* b = memory_pool.b();
        double* c = memory_pool.c();
        double* d = memory_pool.d();
        
        // Set up the tridiagonal system
        for (size_t i = 0; i < n; i++) {
            size_t grid_i = i + 1;  // Offset to interior points
            
            // Pre-calculate thermal diffusivity for better locality
            double kappa = grid.k[grid_i] / (grid.rho[grid_i] * grid.cp[grid_i]);
            double r = dt * kappa;
            
            // Cache grid coefficients for better data locality
            double g1_val = grid.g1[grid_i-1];
            double g2_val = grid.g2[grid_i-1];
            double g_sum = g1_val + g2_val;
            
            // Tridiagonal coefficients (lower, main, upper diagonals)
            a[i] = -0.5 * r * g1_val;
            b[i] = 1.0 + 0.5 * r * g_sum;
            c[i] = -0.5 * r * g2_val;
            
            // Right-hand side - group related operations for better instruction pipelining
            d[i] = T_old[grid_i] + 0.5 * r * (
                     g1_val * T_old[grid_i-1] + 
                     g2_val * T_old[grid_i+1] - 
                     g_sum * T_old[grid_i]);
        }
        
        // Special handling for boundary conditions
        d[0] -= a[0] * grid.T[0];
        d[n-1] -= c[n-1] * grid.T[grid.T.size()-1];
        
        // Run the optimized Thomas algorithm
        solveTridiagonalOptimized(a, b, c, d, n);
        
        // Copy solution back to temperature array (interior points only)
        for (size_t i = 0; i < n; i++) {
            grid.T[i+1] = d[i];
        }
    }
    else {
        // For large systems, use SIMD if available
#if defined(AVX512_ENABLED)
        // AVX-512 path with optimized memory access
        const size_t vec_size = 8;
        
        // Process in blocks of 8 elements using AVX-512
        const size_t vec_end = ((grid.T.size() - 2) / vec_size) * vec_size + 1;
        
        // Create aligned temporary buffers for better performance
        alignas(64) double results[8];
        
        for (size_t i = 1; i < vec_end; i += vec_size) {
            // Bounds check to prevent segfaults
            if (i + vec_size >= grid.T.size() - 1) {
                // Process remaining elements with scalar code
                for (size_t j = i; j < grid.T.size() - 1; j++) {
                    double kappa = grid.k[j] / (grid.rho[j] * grid.cp[j]);
                    double r = dt * kappa;
                    double a = -0.5 * r * grid.g1[j-1];
                    double c = -0.5 * r * grid.g2[j-1];
                    double b = 1.0 + 0.5 * r * (grid.g1[j-1] + grid.g2[j-1]);
                    double rhs_val = T_old[j] + 0.5 * r * (grid.g1[j-1] * T_old[j-1] + 
                                                  grid.g2[j-1] * T_old[j+1] - 
                                                  (grid.g1[j-1] + grid.g2[j-1]) * T_old[j]);
                    grid.T[j] = (rhs_val - a * grid.T[j-1] - c * grid.T[j+1]) / b;
                }
                break;
            }

            // Process 8 elements at once using AVX-512
            __m512d kappa_vec = _mm512_set_pd(
                grid.k[i+7] / (grid.rho[i+7] * grid.cp[i+7]),
                grid.k[i+6] / (grid.rho[i+6] * grid.cp[i+6]),
                grid.k[i+5] / (grid.rho[i+5] * grid.cp[i+5]),
                grid.k[i+4] / (grid.rho[i+4] * grid.cp[i+4]),
                grid.k[i+3] / (grid.rho[i+3] * grid.cp[i+3]),
                grid.k[i+2] / (grid.rho[i+2] * grid.cp[i+2]),
                grid.k[i+1] / (grid.rho[i+1] * grid.cp[i+1]),
                grid.k[i] / (grid.rho[i] * grid.cp[i])
            );
            
            // Compute r = dt * kappa
            __m512d r_vec = _mm512_mul_pd(_mm512_set1_pd(dt), kappa_vec);
            
            // Load grid coefficients - packed for better memory locality
            __m512d g1_vec = _mm512_set_pd(
                grid.g1[i+6], grid.g1[i+5], grid.g1[i+4], grid.g1[i+3],
                grid.g1[i+2], grid.g1[i+1], grid.g1[i], grid.g1[i-1]
            );
            
            __m512d g2_vec = _mm512_set_pd(
                grid.g2[i+6], grid.g2[i+5], grid.g2[i+4], grid.g2[i+3],
                grid.g2[i+2], grid.g2[i+1], grid.g2[i], grid.g2[i-1]
            );
            
            // Constants for vectorized operations
            __m512d half = _mm512_set1_pd(0.5);
            __m512d one = _mm512_set1_pd(1.0);
            __m512d neg_half = _mm512_set1_pd(-0.5);
            
            // Calculate Crank-Nicolson coefficients
            __m512d a_vec = _mm512_mul_pd(neg_half, _mm512_mul_pd(r_vec, g1_vec));
            __m512d c_vec = _mm512_mul_pd(neg_half, _mm512_mul_pd(r_vec, g2_vec));
            
            // g1 + g2 for reuse
            __m512d g_sum_vec = _mm512_add_pd(g1_vec, g2_vec);
            
            // b = 1.0 + 0.5 * r * (g1 + g2)
            __m512d b_vec = _mm512_add_pd(one, _mm512_mul_pd(half, _mm512_mul_pd(r_vec, g_sum_vec)));
            
            // Load temperatures with proper alignment for vectorization
            __m512d T_i_vec = _mm512_set_pd(
                T_old[i+7], T_old[i+6], T_old[i+5], T_old[i+4],
                T_old[i+3], T_old[i+2], T_old[i+1], T_old[i]
            );
            
            __m512d T_im1_vec = _mm512_set_pd(
                T_old[i+6], T_old[i+5], T_old[i+4], T_old[i+3],
                T_old[i+2], T_old[i+1], T_old[i], T_old[i-1]
            );
            
            __m512d T_ip1_vec = _mm512_set_pd(
                T_old[i+8], T_old[i+7], T_old[i+6], T_old[i+5],
                T_old[i+4], T_old[i+3], T_old[i+2], T_old[i+1]
            );
            
            // Compute the explicit part term by term for better instruction pipelining
            // Calculate g1 * T_{i-1}
            __m512d term1 = _mm512_mul_pd(g1_vec, T_im1_vec);
            __m512d term2 = _mm512_mul_pd(g2_vec, T_ip1_vec);
            __m512d term3 = _mm512_mul_pd(g_sum_vec, T_i_vec);
            
            // Combine terms for explicit part: term1 + term2 - term3
            __m512d sum_terms = _mm512_sub_pd(_mm512_add_pd(term1, term2), term3);
            
            // Multiply by 0.5 * r to get explicit part
            __m512d explicit_part = _mm512_mul_pd(half, _mm512_mul_pd(r_vec, sum_terms));
            
            // Calculate RHS: T_old[i] + explicit_part
            __m512d rhs_vec = _mm512_add_pd(T_i_vec, explicit_part);
            
            // Load current boundary temperatures for implicit part
            __m512d T_new_im1_vec = _mm512_set_pd(
                grid.T[i+6], grid.T[i+5], grid.T[i+4], grid.T[i+3],
                grid.T[i+2], grid.T[i+1], grid.T[i], grid.T[i-1]
            );
            
            __m512d T_new_ip1_vec = _mm512_set_pd(
                grid.T[i+8], grid.T[i+7], grid.T[i+6], grid.T[i+5],
                grid.T[i+4], grid.T[i+3], grid.T[i+2], grid.T[i+1]
            );
            
            // Calculate a * T[i-1]
            __m512d a_term = _mm512_mul_pd(a_vec, T_new_im1_vec);
            
            // Calculate c * T[i+1]
            __m512d c_term = _mm512_mul_pd(c_vec, T_new_ip1_vec);
            
            // Compute numerator: rhs - a * T[i-1] - c * T[i+1]
            __m512d numerator = _mm512_sub_pd(_mm512_sub_pd(rhs_vec, a_term), c_term);
            
            // Use reciprocal instead of division for better performance
            __m512d recip_b = _mm512_div_pd(_mm512_set1_pd(1.0), b_vec);
            __m512d result = _mm512_mul_pd(numerator, recip_b);
            
            // Store results in aligned memory
            _mm512_store_pd(results, result);
            
            // Update temperature array with bounds checking
            if (i < grid.T.size() - 1)     grid.T[i] = results[7];
            if (i+1 < grid.T.size() - 1)   grid.T[i+1] = results[6];
            if (i+2 < grid.T.size() - 1)   grid.T[i+2] = results[5];
            if (i+3 < grid.T.size() - 1)   grid.T[i+3] = results[4];
            if (i+4 < grid.T.size() - 1)   grid.T[i+4] = results[3];
            if (i+5 < grid.T.size() - 1)   grid.T[i+5] = results[2];
            if (i+6 < grid.T.size() - 1)   grid.T[i+6] = results[1];
            if (i+7 < grid.T.size() - 1)   grid.T[i+7] = results[0];
        }
        
        // Handle remaining points with scalar code
        for (size_t i = vec_end; i < grid.T.size() - 1; i++) {
            double kappa = grid.k[i] / (grid.rho[i] * grid.cp[i]);
            double r = dt * kappa;
            double a = -0.5 * r * grid.g1[i-1];
            double c = -0.5 * r * grid.g2[i-1];
            double b = 1.0 + 0.5 * r * (grid.g1[i-1] + grid.g2[i-1]);
            double rhs_val = T_old[i] + 0.5 * r * (grid.g1[i-1] * T_old[i-1] + 
                                        grid.g2[i-1] * T_old[i+1] - 
                                        (grid.g1[i-1] + grid.g2[i-1]) * T_old[i]);
            grid.T[i] = (rhs_val - a * grid.T[i-1] - c * grid.T[i+1]) / b;
        }
#elif defined(AVX2_ENABLED)
        // AVX2 implementation (similar but with 4-wide operations)
        const size_t vec_size = 4;
        const size_t vec_end = ((grid.T.size() - 2) / vec_size) * vec_size + 1;
        
        // Create aligned temporary buffer for AVX2
        alignas(32) double results[4];
        
        for (size_t i = 1; i < vec_end; i += vec_size) {
            // Bounds check
            if (i + vec_size >= grid.T.size() - 1) {
                // Process remaining elements with scalar code
                for (size_t j = i; j < grid.T.size() - 1; j++) {
                    double kappa = grid.k[j] / (grid.rho[j] * grid.cp[j]);
                    double r = dt * kappa;
                    double a = -0.5 * r * grid.g1[j-1];
                    double c = -0.5 * r * grid.g2[j-1];
                    double b = 1.0 + 0.5 * r * (grid.g1[j-1] + grid.g2[j-1]);
                    double rhs_val = T_old[j] + 0.5 * r * (grid.g1[j-1] * T_old[j-1] + 
                                                  grid.g2[j-1] * T_old[j+1] - 
                                                  (grid.g1[j-1] + grid.g2[j-1]) * T_old[j]);
                    grid.T[j] = (rhs_val - a * grid.T[j-1] - c * grid.T[j+1]) / b;
                }
                break;
            }

            // Load diffusivity values for 4 elements
            __m256d kappa_vec = _mm256_set_pd(
                grid.k[i+3] / (grid.rho[i+3] * grid.cp[i+3]),
                grid.k[i+2] / (grid.rho[i+2] * grid.cp[i+2]),
                grid.k[i+1] / (grid.rho[i+1] * grid.cp[i+1]),
                grid.k[i] / (grid.rho[i] * grid.cp[i])
            );
            
            // Calculate r = dt * kappa
            __m256d r_vec = _mm256_mul_pd(_mm256_set1_pd(dt), kappa_vec);
            
            // Load grid coefficients
            __m256d g1_vec = _mm256_set_pd(
                grid.g1[i+2], grid.g1[i+1], grid.g1[i], grid.g1[i-1]
            );
            
            __m256d g2_vec = _mm256_set_pd(
                grid.g2[i+2], grid.g2[i+1], grid.g2[i], grid.g2[i-1]
            );
            
            // Constants
            __m256d half = _mm256_set1_pd(0.5);
            __m256d one = _mm256_set1_pd(1.0);
            __m256d neg_half = _mm256_set1_pd(-0.5);
            
            // Calculate coefficients
            __m256d a_vec = _mm256_mul_pd(neg_half, _mm256_mul_pd(r_vec, g1_vec));
            __m256d c_vec = _mm256_mul_pd(neg_half, _mm256_mul_pd(r_vec, g2_vec));
            
            // g1 + g2
            __m256d g_sum_vec = _mm256_add_pd(g1_vec, g2_vec);
            
            // b = 1.0 + 0.5 * r * (g1 + g2)
            __m256d b_vec = _mm256_add_pd(one, _mm256_mul_pd(half, _mm256_mul_pd(r_vec, g_sum_vec)));
            
            // Load temperatures for calculation
            __m256d T_i_vec = _mm256_set_pd(
                T_old[i+3], T_old[i+2], T_old[i+1], T_old[i]
            );
            
            __m256d T_im1_vec = _mm256_set_pd(
                T_old[i+2], T_old[i+1], T_old[i], T_old[i-1]
            );
            
            __m256d T_ip1_vec = _mm256_set_pd(
                T_old[i+4], T_old[i+3], T_old[i+2], T_old[i+1]
            );
            
            // Calculate explicit part
            __m256d term1 = _mm256_mul_pd(g1_vec, T_im1_vec);
            __m256d term2 = _mm256_mul_pd(g2_vec, T_ip1_vec);
            __m256d term3 = _mm256_mul_pd(g_sum_vec, T_i_vec);
            
            // term1 + term2 - term3
            __m256d sum_terms = _mm256_sub_pd(_mm256_add_pd(term1, term2), term3);
            
            // 0.5 * r * (sum_terms)
            __m256d explicit_part = _mm256_mul_pd(half, _mm256_mul_pd(r_vec, sum_terms));
            
            // T_old[i] + explicit_part
            __m256d rhs_vec = _mm256_add_pd(T_i_vec, explicit_part);
            
            // Load current boundary temperatures
            __m256d T_new_im1_vec = _mm256_set_pd(
                grid.T[i+2], grid.T[i+1], grid.T[i], grid.T[i-1]
            );
            
            __m256d T_new_ip1_vec = _mm256_set_pd(
                grid.T[i+4], grid.T[i+3], grid.T[i+2], grid.T[i+1]
            );
            
            // Calculate a * T[i-1]
            __m256d a_term = _mm256_mul_pd(a_vec, T_new_im1_vec);
            
            // Calculate c * T[i+1]
            __m256d c_term = _mm256_mul_pd(c_vec, T_new_ip1_vec);
            
            // rhs - a * T[i-1] - c * T[i+1]
            __m256d numerator = _mm256_sub_pd(_mm256_sub_pd(rhs_vec, a_term), c_term);
            
            // Performance optimization: use reciprocal
            __m256d recip_b = _mm256_div_pd(_mm256_set1_pd(1.0), b_vec);
            __m256d result = _mm256_mul_pd(numerator, recip_b);
            
            // Store results
            _mm256_store_pd(results, result);
            
            // Update temperatures with bounds checking
            if (i < grid.T.size() - 1)     grid.T[i] = results[3];
            if (i+1 < grid.T.size() - 1)   grid.T[i+1] = results[2];
            if (i+2 < grid.T.size() - 1)   grid.T[i+2] = results[1];
            if (i+3 < grid.T.size() - 1)   grid.T[i+3] = results[0];
        }
        
        // Handle remaining points
        for (size_t i = vec_end; i < grid.T.size() - 1; i++) {
            double kappa = grid.k[i] / (grid.rho[i] * grid.cp[i]);
            double r = dt * kappa;
            double a = -0.5 * r * grid.g1[i-1];
            double c = -0.5 * r * grid.g2[i-1];
            double b = 1.0 + 0.5 * r * (grid.g1[i-1] + grid.g2[i-1]);
            double rhs_val = T_old[i] + 0.5 * r * (grid.g1[i-1] * T_old[i-1] + 
                                        grid.g2[i-1] * T_old[i+1] - 
                                        (grid.g1[i-1] + grid.g2[i-1]) * T_old[i]);
            grid.T[i] = (rhs_val - a * grid.T[i-1] - c * grid.T[i+1]) / b;
        }
#else
        // Scalar implementation for systems without AVX
        for (size_t i = 1; i < grid.T.size() - 1; i++) {
            // Pre-calculate all factors to minimize duplicate calculations
            double kappa = grid.k[i] / (grid.rho[i] * grid.cp[i]);
            double r = dt * kappa;
            
            // Cache grid coefficients
            double g1_val = grid.g1[i-1];
            double g2_val = grid.g2[i-1];
            double g_sum = g1_val + g2_val;
            
            // Calculate coefficients only once
            double a = -0.5 * r * g1_val;
            double c = -0.5 * r * g2_val;
            double b = 1.0 + 0.5 * r * g_sum;
            
            // Calculate right-hand side term by term for better instruction pipelining
            double term1 = g1_val * T_old[i-1];
            double term2 = g2_val * T_old[i+1];
            double term3 = g_sum * T_old[i];
            double explicit_part = 0.5 * r * (term1 + term2 - term3);
            
            double rhs_val = T_old[i] + explicit_part;
            
            // Solve for new temperature
            grid.T[i] = (rhs_val - a * grid.T[i-1] - c * grid.T[i+1]) / b;
        }
#endif
    }
}

void surfTemp(Profile& p, double Qs) {
    // Surface parameters
    double epsilon = p.emissivity;
    double sigma = p.config->sigma;
    double k0 = p.k()[0];
    double dz0 = p.dz()[0];
    double T1 = p.T()[1];  // Subsurface temperature
    
    // Physical bounds
    const double MIN_TEMP = 3.0;
    const double MAX_TEMP = 500.0;
    
    // Initial guess for Newton's method
    double Tnew;
    if (Qs > 0) {
        // Use radiative balance for initial guess with input flux
        Tnew = std::pow(Qs / (epsilon * sigma), 0.25);
    } else {
        // Use subsurface temperature as starting point if no input flux
        Tnew = T1;
    }
    
    // Apply Newton-Raphson iteration to solve for surface temperature
    double dT = 1.0;
    int max_iter = 50;
    double tolerance = 0.001;
    double relaxation = 0.8;  // Relaxation factor to improve convergence
    
    for (int iter = 0; iter < max_iter && std::abs(dT) > tolerance; iter++) {
        // Energy balance equation: Qs + k0*(T1-T0)/dz0 = epsilon*sigma*T0^4
        double T0 = Tnew;
        double f = Qs + k0 * (T1 - T0) / dz0 - epsilon * sigma * std::pow(T0, 4);
        
        // Derivative of energy balance equation
        double df = -k0 / dz0 - 4.0 * epsilon * sigma * std::pow(T0, 3);
        
        // Protect against division by zero
        if (std::abs(df) < 1e-10) {
            df = -1e-10;  // Arbitrary small negative value
        }
        
        // Newton-Raphson step with relaxation
        dT = relaxation * f / df;
        Tnew = T0 - dT;
        
        // Apply physical bounds
        Tnew = std::max(Tnew, MIN_TEMP);
        Tnew = std::min(Tnew, MAX_TEMP);
    }
    
    // Final safety bounds check
    if (Tnew < MIN_TEMP) Tnew = MIN_TEMP;
    if (Tnew > MAX_TEMP) Tnew = MAX_TEMP;
    
    p.T()[0] = Tnew;
}

void botTemp(Profile& p, double Qb) {
    // Set bottom temperature using heat flux boundary condition
    int n = p.T().size() - 1;
    double kb = p.k()[n];
    double dzb = p.dz()[n-1];
    
    p.T()[n] = p.T()[n-1] + Qb * dzb / kb;
}

double getTimeStep(const Profile& p, const Planet& planet, double day, 
                   const Config& config, bool verbose) {
    // Calculate maximum stable time step based on von Neumann stability criterion
    // for the Crank-Nicolson scheme (can be up to 0.5)
    double max_fourier = 0.49;  // Close to but less than 0.5 for safety
    
    // Find the most restrictive layer for time stepping
    double dt_min = day;
    int critical_layer = -1;
    double critical_dz = 0.0;
    double critical_kappa = 0.0;
    
    for (size_t i = 0; i < p.z().size(); i++) {
        if (i == 0) continue;  // Skip surface layer
        
        // Calculate thermal diffusivity: kappa = k / (rho * cp)
        double kappa = p.k()[i] / (p.rho()[i] * p.cp()[i]);
        
        // Von Neumann stability condition: dt <= 0.5 * dz^2 / kappa
        double dz = (i < p.z().size() - 1) ? p.dz()[i] : p.dz()[i-1];
        double dt = max_fourier * dz * dz / kappa;
        
        if (dt < dt_min) {
            dt_min = dt;
            critical_layer = i;
            critical_dz = dz;
            critical_kappa = kappa;
        }
    }
    
    // Report on the limiting factor
    if (verbose) {
        std::cout << "Time step limited by layer " << critical_layer 
                  << " (depth=" << p.z()[critical_layer] << "m, dz=" << critical_dz 
                  << "m, kappa=" << critical_kappa << "m²/s)" << std::endl;
        std::cout << "Maximum stable time step: " << dt_min << "s" << std::endl;
    }
    
    // Ensure time step isn't too large or too small
    double min_allowed_dt = 0.1;  // Absolute minimum: 0.1 seconds
    double max_allowed_dt = day / 100.0;  // Ensure at least 100 steps per day
    
    return std::min(std::max(dt_min, min_allowed_dt), max_allowed_dt);
}

// Model implementation
Model::Model(const Planet& planet, double lat, 
             const std::vector<double>& output_times,
             const Eigen::MatrixXd& flux_arr,
             const Config& config,
             bool show_progress) :
    planet(planet),
    lat(lat),
    flux_arr(flux_arr),
    profile(planet, lat, config, show_progress),
    config(&config),
    show_progress(show_progress),
    use_full_time_series(false),
    equilibration_steps(0),
    main_simulation_steps(0) {
    
    // Initialize model
    Qs = 0.0;
    t = 0.0;
    N_z = profile.z().size();
    
    // Store output times
    if (output_times.empty()) {
        // Default: 24 time steps per day
        int steps_per_day = 24;
        double day_fraction = 1.0 / steps_per_day;
        output_time_points.reserve(steps_per_day);
        
        for (int i = 0; i < steps_per_day; i++) {
            output_time_points.push_back(i * day_fraction * planet.day);
        }
    } else {
        // Set output times directly from illumination model
        output_time_points = output_times;
    }
    
    // Set end time based on the last output time
    endtime = output_times.back();
    
    // Calculate number of output steps
    N_steps = output_times.size();
    
    // Extract flux arrays from the input matrix if provided
    if (flux_arr.size() > 0) {
        // Extract time points and flux values from matrix
        flux_times.resize(flux_arr.rows());
        flux_values.resize(flux_arr.rows());
        
        for (int i = 0; i < flux_arr.rows(); i++) {
            flux_times[i] = flux_arr(i, 0);
            flux_values[i] = flux_arr(i, 1);
        }
    }
    
    // Calculate physics-based time step using von Neumann stability criterion
    dt = getTimeStep(profile, planet, planet.day, config, show_progress);
    
    if (show_progress) {
        std::cout << "Model initialized with " << N_z << " layers" << std::endl;
        std::cout << "Time step: " << dt << " s" << std::endl;
        std::cout << "End time: " << endtime << " s" << std::endl;
        std::cout << "Output steps: " << N_steps << std::endl;
    }
}

// Run the model
void Model::run() {
    try {
        // Initialize output arrays
        T = Eigen::MatrixXd::Zero(output_time_points.size(), N_z);
        lt = Eigen::VectorXd::Zero(output_time_points.size());
        
        // Start at the first time point
        t = output_time_points[0];
        
        // Progress bar
        ProgressBar progressBar(output_time_points.size(), "Thermal Simulation", 50, show_progress);
        
        // Pre-equilibrate if needed
        if (config->NYEARSEQ > 0) {
            int equil_days = config->NYEARSEQ * planet.year / planet.day;
            fastEquilibrate(equil_days, t);
        }
        
        // Calculate flux at the first time point
        surfFlux();
        
        // Use a smaller initial time step for stability
        double initial_dt = dt / 10.0;
        
        // Run a few small steps to better align with illumination
        for (int i = 0; i < 10; i++) {
            profile.update_T(initial_dt, Qs, planet.Qb);
            t += initial_dt;
        }
        
        // Reset time to first time point for storage
        t = output_time_points[0];
        
        // Store the first time step
        for (int j = 0; j < N_z; j++) {
            T(0, j) = profile.T()[j];  // Use accessor method
        }
        lt(0) = t;
        
        // Update progress
        progressBar.update(1);
        
        // Process remaining time points
        for (int i = 1; i < output_time_points.size(); i++) {
            double target_time = output_time_points[i];
            
            // Run simulation until we reach the target time
            while (t < target_time) {
                // Calculate flux
                surfFlux();
                
                // Maximum allowable step is distance to next output point
                double step_dt = std::min(dt, target_time - t);
                
                // Update temperature 
                profile.update_T(step_dt, Qs, planet.Qb);
                
                // Update time
                t += step_dt;
            }
            
            // Force exact time alignment
            t = target_time;
            
            // Store results
            for (int j = 0; j < N_z; j++) {
                T(i, j) = profile.T()[j];  // Use accessor method
            }
            lt(i) = target_time;
            
            // Update progress
            progressBar.update(i + 1);
        }
        
        // Ensure progress bar is finished
        progressBar.finish();
    } 
    catch (const std::exception& e) {
        std::cerr << "ERROR in thermal model: " << e.what() << std::endl;
    }
}

void Model::updateOrbit() {
    // Calculate solar angle based on time of day
    // FIXED: Add M_PI to make t=0 correspond to midnight
    double hour_angle = 2.0 * M_PI * std::fmod(t, planet.day) / planet.day + M_PI;
    
    // Normalize to 0-2π range
    hour_angle = std::fmod(hour_angle, 2.0 * M_PI);
    
    // Correct solar zenith angle calculation for equator
    double cos_zenith = std::cos(lat) * std::cos(hour_angle);
    
    // Only apply flux when sun is above horizon
    double flux_factor = std::max(0.0, cos_zenith);
    
    // Calculate absorbed solar flux
    Sabs = planet.S * flux_factor * (1.0 - planet.albedo);
}

// Modify the surfFlux method to use either calculated or interpolated flux
void Model::surfFlux() {
    if (flux_times.empty() || flux_values.empty()) {
        // No flux array provided, use default solar calculation
        updateOrbit();
        Qs = Sabs;
    } else {
        // Use interpolated flux from the provided array - exact time matching
        Qs = interpolateFlux(t);
    }
}

void Model::advance() {
    // Start with a smaller time step for better stability
    double actual_dt = std::min(dt, planet.day / 1000.0);
    
    // Update orbital position
    updateOrbit();
    surfFlux();
    
    // Store old temperatures for monitoring
    std::vector<double> old_temps(profile.T().begin(), profile.T().end());
    
    // Update temperature profile
    profile.update_T(actual_dt, Qs, planet.Qb);
    
    // Stability checking - revert and use smaller step if needed
    double max_surf_change = std::abs(profile.T()[0] - old_temps[0]);
    if (max_surf_change > 5.0) {
        // Excessive change detected - revert and try smaller step
        std::cout << "Excessive temperature change detected (" << max_surf_change 
                  << "K). Using smaller time step." << std::endl;
        
        // Restore old temperatures
        std::copy(old_temps.begin(), old_temps.end(), profile.T().begin());
        
        // Try with 1/4 the time step
        double reduced_dt = actual_dt / 4.0;
        
        // Take 4 smaller steps instead
        for (int i = 0; i < 4; i++) {
            updateOrbit();
            surfFlux();
            profile.update_T(reduced_dt, Qs, planet.Qb);
            t += reduced_dt;
        }
        return;  // Already updated time
    }
    
    // Update time
    t += actual_dt;
}

// Replace the fastEquilibrate method with this properly physics-based version
void Model::fastEquilibrate(int days, double target_time) {
    if (show_progress) {
        std::cout << "Running fast equilibration for " << days << " days..." << std::endl;
    }
    
    // Create progress bar
    ProgressBar eqProgressBar(days, "Fast Equilibration", 50, show_progress);
    
    // Calculate number of steps per day
    int steps_per_day = (int)(planet.day / dt);
    
    // Run equilibration
    for (int day = 0; day < days; day++) {
        eqProgressBar.update(day);
        
        // Process a full day
        for (int step = 0; step < steps_per_day; step++) {
            // Calculate phase within day (0 to 1)
            double day_fraction = (double)step / steps_per_day;
            
            // Calculate hour angle
            double hour_angle = 2.0 * M_PI * day_fraction + M_PI; // Start at midnight
            
            // Calculate zenith angle cosine
            double cos_zenith = std::cos(lat) * std::cos(hour_angle);
            
            // Calculate solar flux
            double flux_factor = std::max(0.0, cos_zenith);
            double local_flux = planet.S * flux_factor * (1.0 - planet.albedo);
            
            // Update temperatures
            profile.update_T(dt, local_flux, planet.Qb);
        }
    }
    
    if (show_progress) {
        eqProgressBar.finish();
        std::cout << "Fast equilibration complete" << std::endl;
    }
    
    // Set time to target time to start main simulation
    t = target_time;
}

// Efficient vector length calculation
inline double vectorLength(const std::array<double, 3>& v) {
    return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

// Optimized dot product calculation
inline double dotProduct(const std::array<double, 3>& v1, const std::array<double, 3>& v2) {
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
}

// Fast calculation of T^3 for radiative conductivity
inline double cube(double x) {
    return x * x * x;
}

// Calculate optimal grid parameters
void calculateOptimalGridParameters(const Planet& planet, Config& config) {
    std::cout << "Optimizing grid parameters..." << std::endl;
    
    // Calculate thermal diffusivity
    double kappa = planet.ks / (planet.rhos * planet.cp0);
    double skin_depth = skinDepth(planet.day, kappa);
    
    // Calculate optimal layers in upper skin depth based on required accuracy
    config.m = std::max(8, static_cast<int>(10.0 * std::sqrt(planet.ks / planet.kd)));
    
    // Calculate optimal growth factor
    config.n = std::max(3, static_cast<int>(20.0 / std::log10(planet.day)));
    
    // Calculate optimal depth in skin depths
    config.b = std::max(10, static_cast<int>(std::sqrt(planet.year / planet.day)));
    
    std::cout << "Optimal grid parameters:" << std::endl;
    std::cout << "  Skin depth layers (m): " << config.m << std::endl;
    std::cout << "  Growth factor (n): " << config.n << std::endl;
    std::cout << "  Total depth in skin depths (b): " << config.b << std::endl;
    std::cout << "  Skin depth: " << skin_depth << " m" << std::endl;
    std::cout << "  Total depth: " << skin_depth * config.b << " m" << std::endl;
}

// Replace the interpolateFlux method with this corrected version
double Model::interpolateFlux(double time) {
    // Handle empty flux array case
    if (flux_times.empty() || flux_values.empty()) {
        return 0.0;
    }
    
    // FIXED: Don't wrap time - use exact time matching instead
    
    // Check if time is outside bounds
    if (time <= flux_times.front()) {
        return flux_values.front();
    }
    
    if (time >= flux_times.back()) {
        return flux_values.back();
    }
    
    // Find the bracketing indices in the time array using binary search
    auto it = std::lower_bound(flux_times.begin(), flux_times.end(), time);
    
    // Calculate interpolation weights
    size_t high_idx = std::distance(flux_times.begin(), it);
    size_t low_idx = high_idx - 1;
    
    double t_low = flux_times[low_idx];
    double t_high = flux_times[high_idx];
    double weight = (time - t_low) / (t_high - t_low);
    
    // Perform linear interpolation
    return flux_values[low_idx] + weight * (flux_values[high_idx] - flux_values[low_idx]);
}

} // namespace thermal