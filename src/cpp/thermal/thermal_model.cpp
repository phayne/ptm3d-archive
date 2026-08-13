#include "thermal_model.h"
#include "../maps/spatial_map.h"
#include "eigen_config.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <atomic>
#include <mutex>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <fstream>
#include <iomanip>
#include <sstream>

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

// Fast power computation helpers
inline double pow2(double x) { return x * x; }
inline double pow3(double x) { return x * x * x; }
inline double pow4(double x) { double x2 = x * x; return x2 * x2; }

// Debug face selection tracking (atomic for thread safety)
static std::atomic<int> debug_face_counter{0};

namespace thermal {

// Thread-local memory pool for Thomas algorithm
thread_local struct {
    std::vector<double> a_vec, b_vec, c_vec, d_vec;
    size_t cached_capacity = 0;  // ADD THIS
    
    void ensure_capacity(size_t n) {
        // Only resize if size actually changed
        if (cached_capacity != n) {
            a_vec.resize(n);
            b_vec.resize(n);
            c_vec.resize(n);
            d_vec.resize(n);
            cached_capacity = n;  // Update cached size
        }
    }
    
    double* a() { return a_vec.data(); }
    double* b() { return b_vec.data(); }
    double* c() { return c_vec.data(); }
    double* d() { return d_vec.data(); }
} memory_pool;

namespace {
inline bool cp_is_temperature_dependent(const Planet& planet) {
    return !planet.cpCoeff.empty();
}
}

// Forward declarations for existing functions
void solveTridiagonal(double* a, double* b, double* c, double* d, int n);
void solveTridiagonalOptimized(double* a, double* b, double* c, double* d, int n);

// Optimized Profile methods
void Profile::update_cp() {
    cp().resize(T().size());
    const size_t n = T().size();

    if (planet.cpCoeff.size() > 0) {
        // Use Horner's method for efficient polynomial evaluation
        // Original form: cp = cp0 + c0*T + c1*T^2 + c2*T^3 + ...
        // Horner form: cp = cp0 + T*(c0 + T*(c1 + T*(c2 + ...)))
        // This reduces 2N multiplications to N multiplications per element
        const size_t deg = planet.cpCoeff.size();
        const double* coeffs = planet.cpCoeff.data();
        const double cp0_val = planet.cp0;

#if defined(AVX512_ENABLED)
        // AVX-512 SIMD Horner's method
        const size_t vec_size = 8;
        __m512d cp0_vec = _mm512_set1_pd(cp0_val);

        #pragma omp parallel for schedule(static) if(n > 1000 && !omp_in_parallel())
        for (size_t i = 0; i < n; i += vec_size) {
            size_t remaining = std::min(vec_size, n - i);

            if (remaining == vec_size) {
                // Full SIMD width - use AVX-512
                __m512d T_vec = _mm512_loadu_pd(&T()[i]);

                // Horner's method with SIMD: start from highest degree coefficient
                __m512d result = _mm512_set1_pd(coeffs[deg - 1]);
                for (int j = static_cast<int>(deg) - 2; j >= 0; --j) {
                    __m512d cj = _mm512_set1_pd(coeffs[j]);
                    result = _mm512_fmadd_pd(result, T_vec, cj);  // result = result*T + coeff[j]
                }
                // cp = cp0 + T * result
                __m512d cp_result = _mm512_fmadd_pd(T_vec, result, cp0_vec);
                _mm512_storeu_pd(&cp()[i], cp_result);
            } else {
                // Handle remainder with scalar code
                for (size_t k = i; k < n; k++) {
                    const double Ti = T()[k];
                    double result = coeffs[deg - 1];
                    for (int j = static_cast<int>(deg) - 2; j >= 0; --j) {
                        result = result * Ti + coeffs[j];
                    }
                    cp()[k] = cp0_val + Ti * result;
                }
            }
        }

#elif defined(AVX2_ENABLED)
        // AVX2 SIMD Horner's method
        const size_t vec_size = 4;
        __m256d cp0_vec = _mm256_set1_pd(cp0_val);

        #pragma omp parallel for schedule(static) if(n > 1000 && !omp_in_parallel())
        for (size_t i = 0; i < n; i += vec_size) {
            size_t remaining = std::min(vec_size, n - i);

            if (remaining == vec_size) {
                // Full SIMD width - use AVX2
                __m256d T_vec = _mm256_loadu_pd(&T()[i]);

                // Horner's method with SIMD: start from highest degree coefficient
                __m256d result = _mm256_set1_pd(coeffs[deg - 1]);
                for (int j = static_cast<int>(deg) - 2; j >= 0; --j) {
                    __m256d cj = _mm256_set1_pd(coeffs[j]);
                    result = _mm256_fmadd_pd(result, T_vec, cj);  // result = result*T + coeff[j]
                }
                // cp = cp0 + T * result
                __m256d cp_result = _mm256_fmadd_pd(T_vec, result, cp0_vec);
                _mm256_storeu_pd(&cp()[i], cp_result);
            } else {
                // Handle remainder with scalar code
                for (size_t k = i; k < n; k++) {
                    const double Ti = T()[k];
                    double result = coeffs[deg - 1];
                    for (int j = static_cast<int>(deg) - 2; j >= 0; --j) {
                        result = result * Ti + coeffs[j];
                    }
                    cp()[k] = cp0_val + Ti * result;
                }
            }
        }

#else
        // Scalar implementation with Horner's method
        #pragma omp parallel for schedule(static) if(n > 1000 && !omp_in_parallel())
        for (size_t i = 0; i < n; i++) {
            const double Ti = T()[i];
            // Horner's method: evaluate from highest to lowest degree
            double result = coeffs[deg - 1];
            for (int j = static_cast<int>(deg) - 2; j >= 0; --j) {
                result = result * Ti + coeffs[j];
            }
            cp()[i] = cp0_val + Ti * result;
        }
#endif
    } else {
        // Constant heat capacity - no temperature dependence
        #pragma omp parallel for schedule(static) if(n > 1000 && !omp_in_parallel())
        for (size_t i = 0; i < n; i++) {
            cp()[i] = planet.cp0;
        }
    }
}

void Profile::update_k() {
    // Hayne et al. (2017) thermal conductivity formula:
    // K = Kc * (1 + chi * (T/350)^3) = Kc * (1 + R350 * T^3)
    // where R350 = chi / 350^3
    k().resize(T().size());
    const size_t n = T().size();

    // Martinez & Siegler (2021) Eq. 10: k = (A1*rho-A2)*k_am(T) + (B1*rho-B2)*T^3.
    // Scalar-only on purpose. k_am(T) is a 9-term Laurent polynomial; hand-
    // vectorizing it would buy little here (this loop is ~10-64 layers) and
    // would add a second numerical expression to keep in step with the OpenCL
    // mirror. Taking the branch OUTSIDE the SIMD blocks also guarantees the
    // Hayne path below is textually and bit-for-bit unchanged.
    if (cond_model == ConductivityModel::Martinez2021) {
        const auto& rho_v = rho();
        #pragma omp parallel for schedule(static) if(n > 1000 && !omp_in_parallel())
        for (size_t i = 0; i < n; i++) {
            k()[i] = martinezK(T()[i], rho_v[i]);
        }
        return;
    }

#if defined(AVX512_ENABLED)
    // AVX-512 implementation with OpenMP
    const size_t vec_size = 8;
    __m512d one = _mm512_set1_pd(1.0);
    __m512d r350 = _mm512_set1_pd(R350_val);

    #pragma omp parallel for schedule(static) if(n > 1000 && !omp_in_parallel())
    for (size_t i = 0; i < n; i += vec_size) {
        size_t remaining = std::min(vec_size, n - i);

        if (remaining == vec_size) {
            // Full SIMD width - use AVX-512
            __m512d T_vec = _mm512_loadu_pd(&T()[i]);
            __m512d kc_vec = _mm512_loadu_pd(&kc()[i]);

            // Calculate T^2 and then T^3
            __m512d T_squared = _mm512_mul_pd(T_vec, T_vec);
            __m512d T_cubed = _mm512_mul_pd(T_squared, T_vec);

            // Calculate (1 + R350 * T^3)
            __m512d factor = _mm512_fmadd_pd(r350, T_cubed, one);

            // Calculate kc * (1 + R350 * T^3)
            __m512d result = _mm512_mul_pd(kc_vec, factor);

            // Store the result
            _mm512_storeu_pd(&k()[i], result);
        } else {
            // Handle remainder with scalar code
            for (size_t j = i; j < n; j++) {
                double t = T()[j];
                k()[j] = kc()[j] * (1.0 + R350_val * (t * t * t));
            }
        }
    }

#elif defined(AVX2_ENABLED)
    // AVX2 implementation with OpenMP
    const size_t vec_size = 4;
    __m256d one = _mm256_set1_pd(1.0);
    __m256d r350 = _mm256_set1_pd(R350_val);

    #pragma omp parallel for schedule(static) if(n > 1000 && !omp_in_parallel())
    for (size_t i = 0; i < n; i += vec_size) {
        size_t remaining = std::min(vec_size, n - i);

        if (remaining == vec_size) {
            // Full SIMD width - use AVX2
            __m256d T_vec = _mm256_loadu_pd(&T()[i]);
            __m256d kc_vec = _mm256_loadu_pd(&kc()[i]);

            // Calculate T^2 and then T^3
            __m256d T_squared = _mm256_mul_pd(T_vec, T_vec);
            __m256d T_cubed = _mm256_mul_pd(T_squared, T_vec);

            // Calculate (1 + R350 * T^3)
            __m256d factor = _mm256_fmadd_pd(r350, T_cubed, one);

            // Calculate kc * (1 + R350 * T^3)
            __m256d result = _mm256_mul_pd(kc_vec, factor);

            // Store the result
            _mm256_storeu_pd(&k()[i], result);
        } else {
            // Handle remainder with scalar code
            for (size_t j = i; j < n; j++) {
                double t = T()[j];
                k()[j] = kc()[j] * (1.0 + R350_val * (t * t * t));
            }
        }
    }

#else
    // Scalar implementation with OpenMP
    // Hayne formula: k = kc * (1 + R350 * T^3)
    #pragma omp parallel for schedule(static) if(n > 1000)
    for (size_t i = 0; i < n; i++) {
        double t = T()[i];
        k()[i] = kc()[i] * (1.0 + R350_val * (t * t * t));
    }
#endif
}

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
    // The Thomas algorithm has strict sequential dependencies (each step depends on
    // the previous step), so cache blocking provides no benefit. The simple algorithm
    // is actually faster due to lower overhead and better branch prediction.
    // Typical thermal profiles have 10-64 layers which fit entirely in L1 cache anyway.
    solveTridiagonal(a, b, c, d, n);
}

// Essential helper functions
double R350(double chi) {
    return chi / pow3(350.0);
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

std::vector<double> generateStableGrid(double skin_depth, int m, int n, int b, bool verbose) {
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
    // sqrt(sqrt(x)) is faster than pow(x, 0.25) and mathematically equivalent
    double ratio = (1.0 - planet.albedo) * solar_flux / (SIGMA_SB * planet.emissivity);
    return std::sqrt(std::sqrt(ratio));
}

double T_eq(const Planet& planet, double lat) {
    double solar_flux = planet.S * std::cos(lat);
    // sqrt(sqrt(x)) is faster than pow(x, 0.25) and mathematically equivalent
    double ratio = (1.0 - planet.albedo) * solar_flux / (SIGMA_SB * planet.emissivity);
    return std::sqrt(std::sqrt(ratio)) / std::sqrt(2.0);
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
    body_radius_km = props.body_radius_km;
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
    body_radius_km = props.body_radius_km;
}

Planet::Planet(const std::string& name, const ThermophysicalProperties& props) {
    this->name = name;
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
    body_radius_km = props.body_radius_km;
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
    double k = ks + 0.01 * pow3(80.0);  // Default conductivity at ~80K
    double rho_val = rhos;  // Default density
    double cp_val = cp0;    // Default heat capacity

    return std::sqrt(k * rho_val * cp_val);
}

double Planet::getAlbedoWithIncidence(double lon, double lat, double incidence_angle) const {
    // Resolve the base normal-incidence albedo A0. If a spatially-varying map
    // was loaded, its per-face value supersedes the body default. The Hayne
    // et al. (2017) angular correction is *still* applied on top — the map
    // describes A0(lon, lat); the (a, b) coefficients describe how A varies
    // with solar zenith angle θ. Returning the bare map value would silently
    // drop the high-θ correction, biasing high-latitude nighttime cooling.
    double A0 = albedo;  // Body-default base albedo
    if (using_albedo_map && albedo_map) {
        double map_value = albedo_map->getValue(lon, lat);
        if (map_value != albedo_map->getMissingValue()) {
            A0 = map_value;
        }
    }
    return albedoWithIncidenceFromA0(A0, incidence_angle);
}

// Per-face variant: A0 supplied directly (e.g. from a loaded per-face albedo
// vector), bypassing the internal spatial-map lookup. Applies the IDENTICAL
// angular model as the (lon, lat) overload, so the CPU mesh flux path can
// compose the incidence albedo on each face's OWN A0 (matching the GPU path)
// without loading a lon/lat SpatialMap onto the thermal planet.
double Planet::getAlbedoWithIncidence(double A0, double incidence_angle) const {
    return albedoWithIncidenceFromA0(A0, incidence_angle);
}

// Shared core: normal-incidence Bond albedo A0 + solar incidence angle ->
// effective visible albedo. Factored out so both getAlbedoWithIncidence
// overloads apply the EXACT same angular model (byte-identical to the
// pre-refactor single-function path).
double Planet::albedoWithIncidenceFromA0(double A0, double incidence_angle) const {
    // Clamp incidence angle to [0, pi/2] - beyond pi/2 is not illuminated
    double theta = std::max(0.0, std::min(incidence_angle, M_PI / 2.0));

    // Directional-hemispheric (Hapke) albedo branch (opt-in, gated). Returns the
    // ANALYTIC closed form A_h(mu0) = 1 - gamma*H(mu0,w), which is guaranteed in
    // [0, 1] for w in [0, 1] and mu0 in [0, 1] (unlike a numerical hemisphere
    // integral of the bidirectional Hapke BRDF, which overshoots 1 at grazing
    // incidence and violates energy conservation). Independent of the Hayne
    // polynomial below, so the baseline path is untouched when the gate is off.
    if (use_hapke_albedo && hapke_albedo_w >= 0.0) {
        // Per-face albedo scaling: anchor the Hapke directional curve to THIS
        // face's Bond albedo A0 (looked up from the map above), not the body-
        // average single-scattering w. gamma = (1-A0)/(1+2*A0) is the exact
        // inverse of A_h(mu0=1)=A0, so A_h(1)=A0 per face; the gate flag
        // (hapke_albedo_w >= 0) still selects the branch. This mirrors the GPU
        // flux path (ptm_thermal_sim hapkeWFromA0 -> w -> gamma=sqrt(1-w)), so
        // CPU==GPU on mapped faces. No-map runs (A0==body albedo everywhere)
        // reduce to the former body-w result -> byte-identical.
        double a0c = std::max(1e-6, std::min(1.0 - 1e-6, A0));
        double gamma = (1.0 - a0c) / (1.0 + 2.0 * a0c);
        double mu0 = std::cos(theta);            // theta already clamped to [0, pi/2]
        mu0 = std::max(0.0, std::min(1.0, mu0));
        double H = (1.0 + 2.0 * mu0) / (1.0 + 2.0 * mu0 * gamma);
        double A_h = 1.0 - gamma * H;
        // Analytic guard: A_h(i) MUST be in [0, 1] for all i and all A0.
        assert(A_h >= -1e-9 && A_h <= 1.0 + 1e-9 &&
               "Hapke A_h(i) out of [0,1] — closed-form energy-conservation guard");
        return std::max(0.0, std::min(1.0, A_h));
    }

    // Check if we have albedo coefficients
    if (albedoCoef.size() >= 2) {
        double a = albedoCoef[0];
        double b = albedoCoef[1];

        // Normalized angles
        double theta_norm_45 = theta / (M_PI / 4.0);  // theta / (pi/4)
        double theta_norm_90 = theta / (M_PI / 2.0);  // theta / (pi/2)

        // Compute albedo with incidence angle correction
        double A = A0 + a * std::pow(theta_norm_45, 3) + b * std::pow(theta_norm_90, 8);

        // Clamp to valid range [0, 1]
        return std::max(0.0, std::min(1.0, A));
    }

    // No coefficients available, return base albedo
    return A0;
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
    else if (prop == "body_radius_km" || prop == "body_radius") body_radius_km = value;
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
Profile::Profile(const Planet& planet, double lat, const Config& config, bool verbose,
                 double H_override) {
    this->planet = planet;
    // If a per-face H override was supplied, propagate it to the local copy of
    // `planet` so that all downstream methods (init_T, update_k, batch grid
    // sharing, etc.) see the right value. The depth-grid initialization below
    // also uses H_face directly to avoid relying on this overwrite for clarity.
    const double H_face = (H_override > 0.0) ? H_override : planet.H;
    this->planet.H = H_face;
    this->config = &config;
    this->emissivity = planet.emissivity;
    this->chi = config.chi;
    this->R350_val = config.getR350();
    this->cond_model = config.conductivity_model;
    this->day = planet.day;

    // Calculate a baseline κ for skin depth using a representative cp value
    // The cp polynomial cp0 + c1*T + c2*T^2 + ... is evaluated at T=200K for a typical estimate
    // This avoids using cp0 directly which may be negative (e.g., -3.6125 for Moon from Hayne et al. 2017)
    double cp_representative = planet.cp0;
    double T_representative = 200.0;  // Representative temperature for cp estimate
    double T_power = T_representative;
    for (size_t j = 0; j < planet.cpCoeff.size(); ++j) {
        cp_representative += planet.cpCoeff[j] * T_power;
        T_power *= T_representative;
    }
    // Ensure cp is positive and reasonable
    if (cp_representative < 100.0) cp_representative = 600.0;  // Fallback to typical lunar regolith value

    double kappa_ref = planet.ks / (planet.rhos * cp_representative);
    double skin_d    = skinDepth(planet.day, kappa_ref);
    if (skin_d < 0.01) skin_d = 0.01; // minimum skin depth

    // Build spatial grid (adaptive or classic). Use this->planet (which has
    // H_face applied) so per-face H affects the depth-grid layout, not just
    // the rho/k initialization below.
    std::vector<double> z_tmp;
    if (config.ADAPTIVE_GRID) {
        z_tmp = generateAdaptiveGrid(this->planet, config, verbose);
    } else {
        z_tmp = generateStableGrid(skin_d, config.m, config.n, config.b, verbose);
    }
    z().assign(z_tmp.begin(), z_tmp.end());
    nlayers = z().size();

    // Derive layer thicknesses. dz is EXACTLY the depth grid's own spacing --
    // there is deliberately no minimum-thickness clamp. That clamp was derived
    // away rather than dropped; the reasoning is recorded here because the
    // constant it replaced looked protective.
    //
    // WHAT THE OLD CLAMP DID. `dz()[i] = max(1e-3, z[i+1]-z[i])` floored dz
    // WITHOUT touching z, so the two described different columns: the
    // conduction stencil (d3z/g1/g2 below, surfTemp, botTemp) integrates dz,
    // while the material model (kc/rho below) and every GPU path index z. The
    // grid then stopped telescoping -- sum(dz) != z_max -- and the CPU stopped
    // matching the GPU, which differences z directly and applies no floor
    // (gpu/thermal_gpu.cpp:319, gpu/thermal_kernels.cl:361). Measured on the
    // shipped asteroid quickstart: dz0 1.559x too thick, 11 layers affected,
    // sum(dz) 2.50% deeper than z_max.
    //
    // WHY NO FLOOR IS NEEDED. The surface node is MASSLESS -- surfTemp solves
    // the algebraic balance Qs + k0(T1-T0)/dz0 = eps*sigma*T0^4 with no
    // rho*c*dz0 capacity -- so the radiative response time
    // rho*c*dz0/(4*eps*sigma*T^3) does not bound the step at all. Of the
    // constraints that remain:
    //   * Surface Newton: its nonlinearity is the top-cell radiative Biot
    //     number Bi = 4*eps*sigma*T^3*dz0/k0, the ratio of the radiative slope
    //     to the conductive slope k0/dz0. Bi -> 0 as dz0 -> 0, so a THINNER
    //     cell makes f more linear and the Newton converges in fewer
    //     iterations. The old floor raised Bi by 1.56x on the asteroid
    //     quickstart -- it made this strictly worse. tools/test_surface_newton
    //     already sweeps dz0 down to 1e-4 m and passes.
    //   * Tridiagonal conditioning: the CN matrix is a strictly diagonally
    //     dominant M-matrix for every dz > 0 and cond ~ 1 + 4*kappa*dt/dz^2.
    //     In float64 that needs dz ~ 1e-10 m to lose 1% -- seven orders below
    //     the old floor. On the GPU the persistent kernel substeps to Fo<0.49,
    //     which bounds its conditioning at ~2 independently of dz.
    //   * float32 depth grid on the GPU: dz must stay above ~10 ulp of z_max,
    //     i.e. ~1.2e-6 * z_max (about 0.11 um for this asteroid grid) -- four
    //     orders below the old floor.
    // What DOES bind is Crank-Nicolson's oscillatory error at the first
    // interior node, governed by the top-cell Fourier number Fo0 =
    // kappa*dt/dz0^2 (amplification (1-Fo0)/(1+Fo0), sign-alternating above
    // 1). Substituting this file's own adaptive grid, dz0 = delta/m with
    // delta = sqrt(2*kappa/omega), gives
    //        Fo0 = pi * m^2 / (steps per rotation),
    // in which kappa CANCELS IDENTICALLY. The binding constraint is therefore
    // dimensionless and couples dt to m; no constant expressed in meters can
    // control it, which is why the floor is gone rather than retuned. The
    // resolution/substep pair is checked explicitly by
    // warnIfTopCellFourierHigh() instead of being silently clamped here.
    dz().resize(z().size() - 1);
    for (int i = 0; i < static_cast<int>(z().size()) - 1; i++) {
        dz()[i] = z()[i+1] - z()[i];
    }

    // Grid coefficients for variable spacing
    d3z().resize(dz().size() - 1);
    g1().resize(dz().size() - 1);
    g2().resize(dz().size() - 1);
    for (int i = 0; i < static_cast<int>(dz().size()) - 1; i++) {
        d3z()[i] = dz()[i+1] * dz()[i] * (dz()[i+1] + dz()[i]);
        g1()[i] = 2.0 * dz()[i+1] / d3z()[i];
        g2()[i] = 2.0 * dz()[i]   / d3z()[i];
    }

    // Initialize thermophysical properties vs. depth.
    // Use the (possibly per-face overridden) H_face stratification scale so
    // that low-H regolith profiles concentrate the conductivity/density
    // transition near the surface and high-H profiles smear it deeper.
    kc().resize(nlayers);
    rho().resize(nlayers);
    for (int i = 0; i < nlayers; i++) {
        double depth = z()[i];
        double df = std::exp(-depth / H_face);
        kc()[i]  = planet.kd + (planet.ks  - planet.kd)  * df;
        rho()[i] = planet.rhod + (planet.rhos - planet.rhod) * df;
    }

    // Initialize temperature, conductivities, cp, kappa
    init_T(planet, lat, verbose);
    update_k();
    update_cp();
    recomputeKappa();

    if (verbose) {
        std::cout << "Initialized thermal profile with " << nlayers << " layers\n"
                  << "Top dz: " << dz().front() << " m, bottom dz: " << dz().back() << " m\n"
                  << "Total depth: " << z().back() << " m\n";
    }
}

void Profile::init_T(const Planet& planet, double lat, bool verbose) {
    // Calculate radiative equilibrium temperature for peak insolation
    // sqrt(sqrt(x)) is faster than pow(x, 0.25) and mathematically equivalent
    double ratio = (1.0 - planet.albedo) * planet.S / (SIGMA_SB * planet.emissivity);
    double T_max = std::sqrt(std::sqrt(ratio));

    // Only print if verbose is true
    if (verbose) {
        std::cout << "Maximum radiative equilibrium temperature: " << T_max << "K" << std::endl;
    }

    // Calculate average temperature (more realistic starting point)
    // For a slowly rotating body like the Moon, average surface T ~ T_max / sqrt(2) for day side
    double T_avg = T_max / std::sqrt(2.0);  // ~270 K for Moon

    // Deep temperature: use Hayne et al. (2017) conductivity formula
    // k_deep = kd * (1 + R350 * T^3) at T ~ T_avg
    double T_deep_estimate = T_avg;
    // Deep conductivity for the geothermal-gradient seed. Under Martinez the
    // deep k depends on rho_d, not on kd (kd does not enter Eq. 10 at all), so
    // using the Hayne expression here would seed a gradient inconsistent with
    // the law actually being integrated.
    double k_deep = (cond_model == ConductivityModel::Martinez2021)
                        ? martinezK(T_deep_estimate, planet.rhod)
                        : planet.kd * (1.0 + R350_val * pow3(T_deep_estimate));
    double gradient = planet.Qb / k_deep;  // K/m
    double depth = z().back();

    // Cap the max temperature increase to prevent runaway initialization
    // For Moon: typical depths are a few meters, gradient ~ 1-3 K/m
    double delta_T_geo = std::min(gradient * depth, 50.0);  // Max 50K geothermal increase
    double T_deep = T_avg + delta_T_geo;

    // Only print if verbose
    if (verbose) {
        std::cout << "Initializing temperature profile:" << std::endl;
        std::cout << "  Surface average: " << T_avg << "K" << std::endl;
        std::cout << "  k_deep: " << k_deep << " W/m/K" << std::endl;
        std::cout << "  Geothermal gradient: " << gradient << " K/m" << std::endl;
        std::cout << "  Grid depth: " << depth << " m" << std::endl;
        std::cout << "  Deep temperature: " << T_deep << "K" << std::endl;
    }

    // Initialize with depth-dependent profile
    T().resize(nlayers);

    // Pre-calculate skin depth once for efficiency
    // Use representative cp at T=200K instead of cp0 (which may be negative for polynomial form)
    double cp_for_skin = planet.cp0;
    double T_repr = 200.0;
    double T_pow = T_repr;
    for (size_t j = 0; j < planet.cpCoeff.size(); ++j) {
        cp_for_skin += planet.cpCoeff[j] * T_pow;
        T_pow *= T_repr;
    }
    if (cp_for_skin < 100.0) cp_for_skin = 600.0;  // Fallback
    double skin_d = skinDepth(planet.day, planet.ks/(planet.rhos*cp_for_skin));
    double inv_skin_d = 1.0 / skin_d;  // Use reciprocal to avoid division in loop

    for (size_t i = 0; i < nlayers; i++) {
        // Exponential blend from surface to deep temperature
        // At surface (z=0): T = T_avg
        // At depth >> skin_d: T approaches T_deep
        double blend = std::exp(-z()[i] * inv_skin_d);
        T()[i] = T_deep + (T_avg - T_deep) * blend;
    }
}

EqSeedResult Profile::seedFromMeanFlux(double mean_absorbed_flux, double Qb,
                                       const EqSeedForcing* forcing,
                                       EqSeedAnchor anchor) {
    EqSeedInput in;
    in.mean_absorbed_flux = mean_absorbed_flux;
    in.Qb                 = Qb;
    in.forcing            = forcing;
    in.anchor             = anchor;
    in.emissivity         = emissivity;
    in.sigma              = config ? config->sigma : SIGMA_SB;
    in.z                  = z().data();
    in.kc                 = kc().data();
    in.rho                = rho().data();
    in.n                  = static_cast<int>(nlayers);
    in.R350               = R350_val;
    in.cond               = cond_model;

    T().resize(nlayers);
    const EqSeedResult r = eqSeedColumn(in, T().data());

    // The constructor does this after init_T; a re-seed must leave the profile
    // in the same state or the first step integrates k/cp/kappa from the OLD
    // temperatures.
    update_k();
    update_cp();
    recomputeKappa();
    return r;
}

// Optimized, race-free, fully vectorized (AVX-512 / AVX2) implementation.
// Features:
//  - Optional fused temperature‑dependent material update (k, cp, kappa) each step
//  - Tridiagonal implicit (Thomas) for small systems
//  - Jacobi (T_old neighbors) large-system path (parallel + SIMD safe)
//  - AVX-512 and AVX2 intrinsics; scalar fallback auto-vectorized with #pragma simd
// Assumptions:
//  - grid.kappa kept consistent externally or recomputed here if temperature dependent
//  - cp polynomial coefficients in planet.cpCoeff (size 0 => constant cp0)
//  - Conductivity k = kc + R350_val * T^3 (if R350_val != 0)
//  - Arrays: grid.T, grid.k, grid.cp, grid.kappa, grid.kc, grid.rho, grid.g1, grid.g2 sized appropriately
// PTM_SOLVER_EQFIX: nodes that the removed T_geo floor WOULD have raised.
// Counted so the release is reported as a number, never assumed to be a no-op.
static std::atomic<long long> g_tgeo_would_clamp{0};
static std::atomic<long long> g_tgeo_tested{0};

void noteTgeoRelease(long long would_clamp, long long tested) {
    if (would_clamp) g_tgeo_would_clamp.fetch_add(would_clamp, std::memory_order_relaxed);
    if (tested)      g_tgeo_tested.fetch_add(tested, std::memory_order_relaxed);
}

// `t_geo_watch` is 0 unless PTM_SOLVER_EQFIX is on AND this face has Qb > 0,
// so the gate-off path pays one compare against a register and touches no
// atomics -- the loop above is unchanged and stays byte-identical.
static inline void countTgeoRelease(const double* T, size_t n, double t_geo_watch) {
    if (t_geo_watch <= 0.0) return;
    long long below = 0;
    for (size_t i = 0; i < n; ++i) if (T[i] < t_geo_watch) ++below;
    noteTgeoRelease(below, static_cast<long long>(n));
}

void Profile::update_T(double dt, double Qs, double Qb) {
    // 0) Boundary conditions (update boundary nodes T[0], T[n-1])
    surfTemp(*this, Qs);
    if (Qb != 0.0) botTemp(*this, Qb);

    // Geothermal-equilibrium floor (mirrors the GPU persistent kernel). In
    // steady darkness the surface radiates the interior heat flux Qb, so no
    // node can sustain a temperature below T_geo = (Qb/(eps*sigma))^0.25. The
    // under-resolved CN substep at a sharp sunlit->shadow transition lets the
    // upper column drift numerically below this physical floor; clamping the
    // finalize loop at T_lo = max(CMB, T_geo) corrects only that sub-floor
    // drift (real PSR floors converge to T_geo from above -> no-op). Per-face,
    // physics-set, NOT the arbitrary [3,500] K clamp.
    //
    // PTM_SOLVER_EQFIX removes it. The floor is not a correction, it is a
    // Dirichlet pin: it holds layer 0 at the analytically exact
    // (Qb/(eps*sigma))^0.25 and thereby MASKS the surface Newton's bias, and
    // its pin on nodes 0/1 defeats the deviation form (measured: arrest onset
    // j=1 with the floor on, none with it off). Only the absolute physical
    // floor (the CMB) survives. Nodes that WOULD have been clamped are counted
    // and reported rather than silently released.
    const double sigma_sb = config ? config->sigma : ptm::kStefanBoltzmann;
    const double T_geo = (Qb > 0.0)
        ? std::sqrt(std::sqrt(Qb / (emissivity * sigma_sb))) : 0.0;
    //
    // Correction-3 rule: under the gate, a path carrying the floor removal
    // WITHOUT the deviation form must refuse, because the two are one physical
    // change. This path does NOT refuse, and the exemption is measured rather
    // than argued: the deviation form's entire benefit is float32 quantization.
    // In the reference measurement (r1_deviation_2026-08-10/arrest_arms2.json)
    // the absolute and deviation forms agree to 6.2e-14 K in float64 -- the
    // algebraic-identity noise floor -- while in float32 the deviation form
    // recovers 0.0649 K of deficit and removes the arrest at j=1. This path
    // stores T in double, so there is no arrest here for the deviation form to
    // remove. The static_assert makes that precondition fail the BUILD rather
    // than the physics if the state type is ever narrowed.
    static_assert(std::is_same<
                      std::remove_reference<decltype(grid.T[0])>::type,
                      double>::value,
                  "Profile state is no longer float64. The PTM_SOLVER_EQFIX "
                  "T_geo-floor removal is exempt from the deviation-form "
                  "requirement ONLY because this path is float64 and cannot "
                  "arrest. Narrowing it re-arms the arrest: implement the R1 "
                  "deviation form here, or make this path refuse under the "
                  "gate like the CPU SIMD and GPU paths do.");
    const bool   eqfix = solverEqFixEnabled();
    const double T_lo  = eqfix ? ABSOLUTE_ZERO_K
                               : std::max(ABSOLUTE_ZERO_K, T_geo);
    const double T_geo_watch = eqfix ? T_geo : 0.0;

    const size_t n_tot = grid.T.size();
    if (n_tot < 3) return;            // need at least 1 interior node
    const size_t n_int = n_tot - 2;   // number of interior nodes

    // 1) Optional fused temperature-dependent material refresh (k, cp, kappa)
    const int mat_interval = (config ? config->MATERIAL_UPDATE_INTERVAL : 1);
    static thread_local size_t s_material_step_counter = 0;
    const bool do_material_refresh = (mat_interval <= 1) || ((s_material_step_counter % mat_interval) == 0);

    // Martinez & Siegler (2021) makes k temperature-dependent through the
    // SOLID term, so it is temperature-dependent even when R350 == 0.
    const bool martinez    = (cond_model == ConductivityModel::Martinez2021);
    const bool temp_dep_k  = martinez || (R350_val != 0.0);
    const bool temp_dep_cp = !planet.cpCoeff.empty();

    // Martinez path: scalar, kept out of the SIMD block below so that block
    // stays textually and bit-for-bit identical for the default Hayne law.
    if (martinez && do_material_refresh) {
        const size_t deg = planet.cpCoeff.size();
        for (size_t i = 0; i < n_tot; ++i) {
            const double Ti = grid.T[i];
            grid.k[i] = martinezK(Ti, grid.rho[i]);
            if (temp_dep_cp) {
                double result = planet.cpCoeff[deg - 1];
                for (int j = static_cast<int>(deg) - 2; j >= 0; --j) {
                    result = result * Ti + planet.cpCoeff[j];
                }
                grid.cp[i] = planet.cp0 + Ti * result;
            }
            const double cpused = temp_dep_cp ? grid.cp[i] : planet.cp0;
            grid.kappa[i] = grid.k[i] / (grid.rho[i] * cpused);
        }
    }

    if (!martinez && (temp_dep_k || temp_dep_cp) && do_material_refresh) {
        const size_t n = n_tot;
    #if defined(AVX512_ENABLED)
        const size_t VS = 8;
        size_t i = 0;
        const size_t deg = planet.cpCoeff.size();
        __m512d r350 = _mm512_set1_pd(R350_val);
        __m512d cp0v = _mm512_set1_pd(planet.cp0);
        for (; i + VS <= n; i += VS) {
            __m512d Tvec  = _mm512_loadu_pd(&grid.T[i]);
            // k(T) = kc * (1 + R350*T^3)  -- Hayne et al. (2017) formula
            if (temp_dep_k) {
                __m512d kcvec = _mm512_loadu_pd(&grid.kc[i]);
                __m512d t2    = _mm512_mul_pd(Tvec, Tvec);
                __m512d t3    = _mm512_mul_pd(t2, Tvec);
                __m512d one   = _mm512_set1_pd(1.0);
                __m512d factor = _mm512_fmadd_pd(r350, t3, one);  // 1 + R350*T^3
                __m512d kvec  = _mm512_mul_pd(kcvec, factor);
                _mm512_storeu_pd(&grid.k[i], kvec);
            }
            // cp(T) = cp0 + T * (c0 + T * (c1 + ...)) using Horner's method
            __m512d cpv = cp0v;
            if (temp_dep_cp) {
                // Horner's method: start from highest degree coefficient
                __m512d result = _mm512_set1_pd(planet.cpCoeff[deg - 1]);
                for (int j = static_cast<int>(deg) - 2; j >= 0; --j) {
                    __m512d cj = _mm512_set1_pd(planet.cpCoeff[j]);
                    result = _mm512_fmadd_pd(result, Tvec, cj);  // result = result*T + coeff[j]
                }
                // cp = cp0 + T * result
                cpv = _mm512_fmadd_pd(Tvec, result, cp0v);
                _mm512_storeu_pd(&grid.cp[i], cpv);
            }
            // kappa = k / (rho * cp)
            __m512d kval   = _mm512_loadu_pd(&grid.k[i]);
            __m512d rhov   = _mm512_loadu_pd(&grid.rho[i]);
            __m512d cpuse  = temp_dep_cp ? _mm512_loadu_pd(&grid.cp[i]) : cpv;
            __m512d denom  = _mm512_mul_pd(rhov, cpuse);
            __m512d kappav = _mm512_div_pd(kval, denom);
            _mm512_storeu_pd(&grid.kappa[i], kappav);
        }
        for (; i < n; ++i) {
            if (temp_dep_k) {
                const double t = grid.T[i];
                grid.k[i] = grid.kc[i] * (1.0 + R350_val * (t * t * t));
            }
            if (temp_dep_cp) {
                // Horner's method for cp polynomial
                const double Ti = grid.T[i];
                double result = planet.cpCoeff[deg - 1];
                for (int j = static_cast<int>(deg) - 2; j >= 0; --j) {
                    result = result * Ti + planet.cpCoeff[j];
                }
                grid.cp[i] = planet.cp0 + Ti * result;
            }
            const double cpused = temp_dep_cp ? grid.cp[i] : planet.cp0;
            grid.kappa[i] = grid.k[i] / (grid.rho[i] * cpused);
        }
    #elif defined(AVX2_ENABLED)
        const size_t VS = 4;
        size_t i = 0;
        const size_t deg = planet.cpCoeff.size();
        __m256d r350 = _mm256_set1_pd(R350_val);
        __m256d cp0v = _mm256_set1_pd(planet.cp0);
        for (; i + VS <= n; i += VS) {
            __m256d Tvec  = _mm256_loadu_pd(&grid.T[i]);
            // k(T) = kc * (1 + R350*T^3)  -- Hayne et al. (2017) formula
            if (temp_dep_k) {
                __m256d kcvec = _mm256_loadu_pd(&grid.kc[i]);
                __m256d t2    = _mm256_mul_pd(Tvec, Tvec);
                __m256d t3    = _mm256_mul_pd(t2, Tvec);
                __m256d one   = _mm256_set1_pd(1.0);
                __m256d factor = _mm256_fmadd_pd(r350, t3, one);  // 1 + R350*T^3
                __m256d kvec  = _mm256_mul_pd(kcvec, factor);
                _mm256_storeu_pd(&grid.k[i], kvec);
            }
            // cp(T) = cp0 + T * (c0 + T * (c1 + ...)) using Horner's method
            __m256d cpv = cp0v;
            if (temp_dep_cp) {
                // Horner's method: start from highest degree coefficient
                __m256d result = _mm256_set1_pd(planet.cpCoeff[deg - 1]);
                for (int j = static_cast<int>(deg) - 2; j >= 0; --j) {
                    __m256d cj = _mm256_set1_pd(planet.cpCoeff[j]);
                    result = _mm256_fmadd_pd(result, Tvec, cj);  // result = result*T + coeff[j]
                }
                // cp = cp0 + T * result
                cpv = _mm256_fmadd_pd(Tvec, result, cp0v);
                _mm256_storeu_pd(&grid.cp[i], cpv);
            }
            __m256d kval   = _mm256_loadu_pd(&grid.k[i]);
            __m256d rhov   = _mm256_loadu_pd(&grid.rho[i]);
            __m256d cpuse  = temp_dep_cp ? _mm256_loadu_pd(&grid.cp[i]) : cpv;
            __m256d denom  = _mm256_mul_pd(rhov, cpuse);
            __m256d kappav = _mm256_div_pd(kval, denom);
            _mm256_storeu_pd(&grid.kappa[i], kappav);
        }
        for (; i < n; ++i) {
            if (temp_dep_k) {
                const double t = grid.T[i];
                grid.k[i] = grid.kc[i] * (1.0 + R350_val * (t * t * t));
            }
            if (temp_dep_cp) {
                // Horner's method for cp polynomial
                const double Ti = grid.T[i];
                double result = planet.cpCoeff[deg - 1];
                for (int j = static_cast<int>(deg) - 2; j >= 0; --j) {
                    result = result * Ti + planet.cpCoeff[j];
                }
                grid.cp[i] = planet.cp0 + Ti * result;
            }
            const double cpused = temp_dep_cp ? grid.cp[i] : planet.cp0;
            grid.kappa[i] = grid.k[i] / (grid.rho[i] * cpused);
        }
    #else
        // Scalar fallback with Horner's method for cp
        const size_t deg = planet.cpCoeff.size();
        for (size_t i = 0; i < n; ++i) {
            if (temp_dep_k) {
                const double t = grid.T[i];
                grid.k[i] = grid.kc[i] * (1.0 + R350_val * (t * t * t));
            }
            if (temp_dep_cp) {
                // Horner's method for cp polynomial
                const double Ti = grid.T[i];
                double result = planet.cpCoeff[deg - 1];
                for (int j = static_cast<int>(deg) - 2; j >= 0; --j) {
                    result = result * Ti + planet.cpCoeff[j];
                }
                grid.cp[i] = planet.cp0 + Ti * result;
            }
            const double cpused = temp_dep_cp ? grid.cp[i] : planet.cp0;
            grid.kappa[i] = grid.k[i] / (grid.rho[i] * cpused);
        }
    #endif
    }
    ++s_material_step_counter;

    // 2) Snapshot old temperatures (thread-local scratch)
    alignas(64) static thread_local std::vector<double> T_old;
    static thread_local size_t cached_n = 0;
    if (cached_n != n_tot) {
        T_old.resize(n_tot);
        cached_n = n_tot;
    }
    std::copy(grid.T.begin(), grid.T.end(), T_old.begin());

    // Aliases (restrict to help vectorizer)
    double* __restrict Tptr         = grid.T.data();
    const double* __restrict kappa  = grid.kappa.data();
    const double* __restrict g1p    = grid.g1.data();   // size n_int (coupling i-1->i)
    const double* __restrict g2p    = grid.g2.data();   // size n_int (coupling i+1->i)

    // 3) Decide solver path: implicit-for-all (Crank–Nicolson + Thomas) or fallback
    bool use_implicit = false;
    if (config) {
        if (config->IMPLICIT_ALL) {
            use_implicit = true;
        } else if (n_int <= static_cast<size_t>(std::max(1, config->THOMAS_LIMIT))) {
            use_implicit = true;
        }
    }

    if (use_implicit) {
        // Assemble tridiagonal CN system A x = d for interior nodes [1..n_tot-2]
        // Using flux-based formulation: dT/dt = (1/(rho*cp)) * d/dz(k * dT/dz)
        // This properly handles spatially-varying conductivity by using k at interfaces
        if (n_int == 0) return;

        memory_pool.ensure_capacity(n_int);
        double* a = memory_pool.a();  // lower diag
        double* b = memory_pool.b();  // main diag
        double* c = memory_pool.c();  // upper diag
        double* d = memory_pool.d();  // RHS (will be overwritten with solution)

        const double* __restrict kptr = grid.k.data();
        const double* __restrict rhoptr = grid.rho.data();
        const double* __restrict cpptr = grid.cp.data();

        #pragma omp simd
        for (size_t j = 0; j < n_int; ++j) {
            const size_t i   = j + 1;       // interior node index in [1..n_tot-2]
            const double g1v = g1p[i-1];    // coefficient for T[i-1] coupling
            const double g2v = g2p[i-1];    // coefficient for T[i+1] coupling

            // Flux-based formulation: use k[i-1] for lower interface, k[i] for upper
            // This matches heat1d and properly handles variable conductivity
            const double alpha = g1v * kptr[i-1];  // conductivity-weighted lower coupling
            const double beta  = g2v * kptr[i];    // conductivity-weighted upper coupling
            const double inv_rhocp = dt / (rhoptr[i] * cpptr[i]);

            a[j] = -0.5 * inv_rhocp * alpha;
            b[j] =  1.0 + 0.5 * inv_rhocp * (alpha + beta);
            c[j] = -0.5 * inv_rhocp * beta;

            // RHS using old neighbors (Crank-Nicolson explicit part)
            d[j] = T_old[i] + 0.5 * inv_rhocp * (alpha * T_old[i-1] - (alpha + beta) * T_old[i] + beta * T_old[i+1]);
        }

        // Fold boundary values into RHS and zero the corresponding coefficients
        d[0]       -= a[0] * Tptr[0];
        d[n_int-1] -= c[n_int-1] * Tptr[n_tot - 1];
        a[0] = 0.0;
        c[n_int - 1] = 0.0;

        // Light diagonal regularization to avoid accidental zero pivots
        constexpr double eps = 1e-15;
        #pragma omp simd
        for (size_t j = 0; j < n_int; ++j) {
            if (!std::isfinite(b[j]) || std::abs(b[j]) < eps) {
                b[j] = (b[j] >= 0.0 ? eps : -eps);
            }
        }

        // Solve (solution overwrites d)
        solveTridiagonalOptimized(a, b, c, d, static_cast<int>(n_int));

        // Scatter back into interior nodes
        #pragma omp simd
        for (size_t j = 0; j < n_int; ++j) {
            Tptr[j + 1] = d[j];
        }

        // Physical clamping and NaN guard
        #pragma omp parallel for schedule(static) if(n_tot > 1024)
        for (size_t i = 0; i < n_tot; ++i) {
            if (!std::isfinite(Tptr[i])) Tptr[i] = T_old[i];
            Tptr[i] = std::max(T_lo, std::min(MAX_PHYSICAL_TEMP_K, Tptr[i]));
        }
        countTgeoRelease(Tptr, n_tot, T_geo_watch);
        return;
    }

    // 4) Fallback paths:
    //    - If small enough, use Thomas as before
    //    - Else, do a single CN Jacobi-like sweep with SIMD (uses T_old neighbors)

    if (n_int <= THOMAS_MAX_SIZE) {
        // Small system fallback: Thomas algorithm with flux-based formulation
        memory_pool.ensure_capacity(n_int);
        double* a = memory_pool.a();
        double* b = memory_pool.b();
        double* c = memory_pool.c();
        double* d = memory_pool.d();

        const double* __restrict kptr = grid.k.data();
        const double* __restrict rhoptr = grid.rho.data();
        const double* __restrict cpptr = grid.cp.data();

        #pragma omp simd
        for (size_t j = 0; j < n_int; ++j) {
            const size_t i   = j + 1;
            const double g1v = g1p[i-1];
            const double g2v = g2p[i-1];

            // Flux-based formulation
            const double alpha = g1v * kptr[i-1];
            const double beta  = g2v * kptr[i];
            const double inv_rhocp = dt / (rhoptr[i] * cpptr[i]);

            a[j] = -0.5 * inv_rhocp * alpha;
            b[j] =  1.0 + 0.5 * inv_rhocp * (alpha + beta);
            c[j] = -0.5 * inv_rhocp * beta;

            d[j] = T_old[i] + 0.5 * inv_rhocp * (alpha * T_old[i-1] - (alpha + beta) * T_old[i] + beta * T_old[i+1]);
        }

        d[0]       -= a[0] * Tptr[0];
        d[n_int-1] -= c[n_int-1] * Tptr[n_tot - 1];
        a[0] = 0.0;
        c[n_int - 1] = 0.0;

        solveTridiagonalOptimized(a, b, c, d, static_cast<int>(n_int));

        #pragma omp simd
        for (size_t j = 0; j < n_int; ++j) {
            Tptr[j + 1] = d[j];
        }

        #pragma omp parallel for schedule(static) if(n_tot > 1024)
        for (size_t i = 0; i < n_tot; ++i) {
            if (!std::isfinite(Tptr[i])) Tptr[i] = T_old[i];
            Tptr[i] = std::max(T_lo, std::min(MAX_PHYSICAL_TEMP_K, Tptr[i]));
        }
        countTgeoRelease(Tptr, n_tot, T_geo_watch);
        return;
    }

    // Large system: SIMD-friendly single CN sweep (Jacobi update using T_old neighbors)
    // Large system fallback: use flux-based explicit update (no SIMD for now to ensure correctness)
    // The SIMD versions below use the old kappa-based formulation and need rewriting for flux-based
    const double* __restrict kptr = grid.k.data();
    const double* __restrict rhoptr = grid.rho.data();
    const double* __restrict cpptr = grid.cp.data();

    #pragma omp parallel for schedule(static) if(!omp_in_parallel())
    for (size_t i = 1; i < n_tot - 1; ++i) {
        const double g1v  = g1p[i-1];
        const double g2v  = g2p[i-1];

        // Flux-based formulation
        const double alpha = g1v * kptr[i-1];
        const double beta  = g2v * kptr[i];
        const double inv_rhocp = dt / (rhoptr[i] * cpptr[i]);

        // Explicit forward Euler with flux-based coefficients
        Tptr[i] = T_old[i] + inv_rhocp * (alpha * T_old[i-1] - (alpha + beta) * T_old[i] + beta * T_old[i+1]);
    }

#if 0  // Disabled: SIMD paths need updating for flux-based formulation
#if defined(AVX512_ENABLED)
    {
        const size_t VS = 8;
        const size_t last = n_tot - 2;
        size_t i = 1;
        __m512d half = _mm512_set1_pd(0.5);
        __m512d one  = _mm512_set1_pd(1.0);
        __m512d dtv  = _mm512_set1_pd(dt);

        #pragma omp parallel if(!omp_in_parallel())
        {
            #pragma omp for schedule(static)
            for (i = 1; i + VS - 1 <= last; i += VS) {
                __m512d Told_im1 = _mm512_loadu_pd(&T_old[i-1]);
                __m512d Told_i   = _mm512_loadu_pd(&T_old[i]);
                __m512d Told_ip1 = _mm512_loadu_pd(&T_old[i+1]);

                __m512d kappav = _mm512_loadu_pd(&kappa[i]);
                __m512d r      = _mm512_mul_pd(dtv, kappav);

                __m512d g1v = _mm512_loadu_pd(&g1p[i-1]);
                __m512d g2v = _mm512_loadu_pd(&g2p[i-1]);
                __m512d sumg = _mm512_add_pd(g1v, g2v);

                __m512d half_r = _mm512_mul_pd(half, r);
                __m512d a = _mm512_mul_pd(_mm512_set1_pd(-1.0), _mm512_mul_pd(half_r, g1v));
                __m512d c = _mm512_mul_pd(_mm512_set1_pd(-1.0), _mm512_mul_pd(half_r, g2v));
                __m512d b = _mm512_add_pd(one, _mm512_mul_pd(half_r, sumg));

                __m512d term_left   = _mm512_mul_pd(g1v, Told_im1);
                __m512d term_right  = _mm512_mul_pd(g2v, Told_ip1);
                __m512d term_center = _mm512_mul_pd(sumg, Told_i);
                __m512d bracket     = _mm512_sub_pd(_mm512_add_pd(term_left, term_right), term_center);
                __m512d rhs         = _mm512_add_pd(Told_i, _mm512_mul_pd(half_r, bracket));

                __m512d num = _mm512_sub_pd(
                                  _mm512_sub_pd(rhs, _mm512_mul_pd(a, Told_im1)),
                                  _mm512_mul_pd(c, Told_ip1));
                __m512d Tnew = _mm512_div_pd(num, b);
                _mm512_storeu_pd(&Tptr[i], Tnew);
            }
        }
        for (; i <= last; ++i) {
            const double r    = dt * kappa[i];
            const double g1v  = g1p[i-1];
            const double g2v  = g2p[i-1];
            const double sumg = g1v + g2v;
            const double a    = -0.5 * r * g1v;
            const double c    = -0.5 * r * g2v;
            const double b    =  1.0 + 0.5 * r * sumg;
            const double rhs  =  T_old[i] + 0.5 * r * (g1v * T_old[i-1] + g2v * T_old[i+1] - sumg * T_old[i]);
            const double num  = rhs - a * T_old[i-1] - c * T_old[i+1];
            Tptr[i] = num / b;
        }
    }
#elif defined(AVX2_ENABLED)
    {
        const size_t VS = 4;
        const size_t last = n_tot - 2;
        size_t i = 1;
        __m256d half = _mm256_set1_pd(0.5);
        __m256d one  = _mm256_set1_pd(1.0);
        __m256d dtv  = _mm256_set1_pd(dt);

        #pragma omp parallel if(!omp_in_parallel())
        {
            #pragma omp for schedule(static)
            for (i = 1; i + VS - 1 <= last; i += VS) {
                __m256d Told_im1 = _mm256_loadu_pd(&T_old[i-1]);
                __m256d Told_i   = _mm256_loadu_pd(&T_old[i]);
                __m256d Told_ip1 = _mm256_loadu_pd(&T_old[i+1]);

                __m256d kappav = _mm256_loadu_pd(&kappa[i]);
                __m256d r      = _mm256_mul_pd(dtv, kappav);

                __m256d g1v = _mm256_loadu_pd(&g1p[i-1]);
                __m256d g2v = _mm256_loadu_pd(&g2p[i-1]);
                __m256d sumg = _mm256_add_pd(g1v, g2v);

                __m256d half_r = _mm256_mul_pd(half, r);
                __m256d a = _mm256_mul_pd(_mm256_set1_pd(-1.0), _mm256_mul_pd(half_r, g1v));
                __m256d c = _mm256_mul_pd(_mm256_set1_pd(-1.0), _mm256_mul_pd(half_r, g2v));
                __m256d b = _mm256_add_pd(one, _mm256_mul_pd(half_r, sumg));

                __m256d term_left   = _mm256_mul_pd(g1v, Told_im1);
                __m256d term_right  = _mm256_mul_pd(g2v, Told_ip1);
                __m256d term_center = _mm256_mul_pd(sumg, Told_i);
                __m256d bracket     = _mm256_sub_pd(_mm256_add_pd(term_left, term_right), term_center);
                __m256d rhs         = _mm256_add_pd(Told_i, _mm256_mul_pd(half_r, bracket));

                __m256d num = _mm256_sub_pd(
                                  _mm256_sub_pd(rhs, _mm256_mul_pd(a, Told_im1)),
                                  _mm256_mul_pd(c, Told_ip1));
                __m256d Tnew = _mm256_div_pd(num, b);
                _mm256_storeu_pd(&Tptr[i], Tnew);
            }
        }
        for (; i <= last; ++i) {
            const double r    = dt * kappa[i];
            const double g1v  = g1p[i-1];
            const double g2v  = g2p[i-1];
            const double sumg = g1v + g2v;
            const double a    = -0.5 * r * g1v;
            const double c    = -0.5 * r * g2v;
            const double b    =  1.0 + 0.5 * r * sumg;
            const double rhs  =  T_old[i] + 0.5 * r * (g1v * T_old[i-1] + g2v * T_old[i+1] - sumg * T_old[i]);
            const double num  = rhs - a * T_old[i-1] - c * T_old[i+1];
            Tptr[i] = num / b;
        }
    }
#else
    #pragma omp parallel for schedule(static) if(!omp_in_parallel())
    for (size_t i = 1; i < n_tot - 1; ++i) {
        const double r    = dt * kappa[i];
        const double g1v  = g1p[i-1];
        const double g2v  = g2p[i-1];
        const double sumg = g1v + g2v;
        const double a    = -0.5 * r * g1v;
        const double c    = -0.5 * r * g2v;
        const double b    =  1.0 + 0.5 * r * sumg;
        const double rhs  =  T_old[i] + 0.5 * r * (g1v * T_old[i-1] + g2v * T_old[i+1] - sumg * T_old[i]);
        const double num  = rhs - a * T_old[i-1] - c * T_old[i+1];
        Tptr[i] = num / b;
    }
#endif  // AVX512/AVX2/scalar
#endif  // #if 0 - disabled SIMD paths

    // Clamp for safety on fallback path
    #pragma omp parallel for schedule(static) if(n_tot > 1024)
    for (size_t i = 0; i < n_tot; ++i) {
        if (!std::isfinite(Tptr[i])) Tptr[i] = T_old[i];
        Tptr[i] = std::max(T_lo, std::min(MAX_PHYSICAL_TEMP_K, Tptr[i]));
    }
    countTgeoRelease(Tptr, n_tot, T_geo_watch);
}

// See the declaration in thermal_model.h for why the three changes share one
// gate. Announced once so a run's log records which solver it used.
bool solverEqFixEnabled() {
    static const bool on = [] {
        const char* e = std::getenv("PTM_SOLVER_EQFIX");
        const bool v = (e && *e && std::strcmp(e, "0") != 0);
        if (v) {
            std::cout << "[Thermal] PTM_SOLVER_EQFIX=1: T_geo-floor release "
                         "ENABLED (non-default solver). NOTE: the surface-Newton "
                         "exit criterion is no longer part of this gate -- it is "
                         "unconditional on every path as of "
                         "fix/surface-bc-unification." << std::endl;
        }
        return v;
    }();
    return on;
}

// Surface-BC certificate, accumulated PER THREAD.
//
// It records the per-run maximum achieved |f|/(4*eps*sigma*T^3), the mean
// iteration count, and three exception counts: the step limiter's growth
// branch (unreachable from the max(T_eq, T1) seed, so if it fires that premise
// has broken and the iterate can be off by hundreds of K), the iteration cap,
// and solves that only the round-off floor could terminate.
//
// Deliberately NOT atomics. The surface BC runs once per face per substep:
// 3.8e8 solves on a 12,800-face lunation at hourly cadence. Two contended
// relaxed fetch_adds per solve cost 8.5x the ENTIRE thermal phase -- measured
// 2 s -> 17 s on crater80N, with the scalar surface solve itself costing
// nothing measurable. Each thread owns its block; reportSurfaceNewtonStats()
// sums them once, at the end.
namespace {
struct SurfaceBCTally {
    long long solves = 0, iters = 0, ulp = 0, growth = 0, cap = 0;
    double    max_resid = 0.0;
};
std::mutex g_tally_mutex;
std::vector<SurfaceBCTally*> g_tallies;

// Deliberately never freed: the registry must outlive every worker thread, and
// it is one 48-byte block per thread.
SurfaceBCTally* newSurfaceBCTally() {
    auto* t = new SurfaceBCTally();
    std::lock_guard<std::mutex> lk(g_tally_mutex);
    g_tallies.push_back(t);
    return t;
}
}  // namespace

double surfaceBCTolK() {
    static const double tol = [] {
        // Legacy name honored, with a warning, so an existing diagnostic script
        // does not silently lose its override.
        const char* legacy = std::getenv("PTM_SOLVER_EQFIX_TOLK");
        const char* e      = std::getenv("PTM_SURFACE_BC_TOLK");
        if ((!e || !*e) && legacy && *legacy) {
            std::cout << "[Thermal] PTM_SOLVER_EQFIX_TOLK is deprecated (the "
                         "criterion is no longer gated); use "
                         "PTM_SURFACE_BC_TOLK. Honoring it for this run."
                      << std::endl;
            e = legacy;
        }
        if (!e || !*e) return kSurfaceBCTolK;
        const double v = std::atof(e);
        if (!(v > 0.0)) {
            std::cerr << "[Thermal] WARNING: PTM_SURFACE_BC_TOLK='" << e
                      << "' is not a positive number; keeping the derived "
                      << kSurfaceBCTolK << " K." << std::endl;
            return kSurfaceBCTolK;
        }
        std::cout << "[Thermal] PTM_SURFACE_BC_TOLK=" << v
                  << " K overrides the derived surface-BC tolerance ("
                  << kSurfaceBCTolK << " K). Diagnostic use only." << std::endl;
        return v;
    }();
    return tol;
}

void noteSurfaceBCSolution(const SurfaceBCSolution& s) {
    static thread_local SurfaceBCTally* t = newSurfaceBCTally();
    ++t->solves;
    t->iters += s.iterations;
    if (s.growth_capped) ++t->growth;
    if (s.hit_iter_cap)  ++t->cap;
    if (s.ulp_limited)   ++t->ulp;
    if (s.residual_K > t->max_resid) t->max_resid = s.residual_K;
}

void reportSurfaceNewtonStats() {
    long long g = 0, c = 0, ns = 0, ni = 0, nu = 0;
    double rm = 0.0;
    {
        std::lock_guard<std::mutex> lk(g_tally_mutex);
        for (const SurfaceBCTally* t : g_tallies) {
            g  += t->growth;
            c  += t->cap;
            ns += t->solves;
            ni += t->iters;
            nu += t->ulp;
            if (t->max_resid > rm) rm = t->max_resid;
        }
    }
    const long long w = g_tgeo_would_clamp.load(std::memory_order_relaxed);
    const long long n = g_tgeo_tested.load(std::memory_order_relaxed);
    // Printed through a LOCAL stream. std::cout arrives here carrying
    // `std::fixed << std::setprecision(1)` from an earlier banner, which
    // renders a 3e-13 K residual as "0.0" -- a run reporting a number that is
    // not the number it measured, i.e. the PTM-05/PTM-55 archetype.
    if (ns > 0) {
        std::ostringstream o;
        o << "[Thermal] surface-BC certificate: max achieved "
             "|f|/(4 eps sigma T^3) = " << std::scientific << std::setprecision(3)
          << rm << " K over " << std::defaultfloat << ns
          << " solves (tol_K = " << std::scientific << std::setprecision(1)
          << surfaceBCTolK() << " K), mean " << std::fixed << std::setprecision(2)
          << (static_cast<double>(ni) / static_cast<double>(ns))
          << " Newton iterations/solve.";
        std::cout << o.str() << std::endl;
    }
    if (nu > 0) {
        std::ostringstream o;
        o << "[Thermal] surface-BC round-off floor bound on " << nu << " / " << ns
          << " solves (" << std::fixed << std::setprecision(3)
          << (100.0 * static_cast<double>(nu) / static_cast<double>(ns))
          << "%): those iterates are at the precision limit of the residual "
             "test, not at tol_K. Reported, not hidden.";
        std::cout << o.str() << std::endl;
    }
    if (n > 0) {
        std::cout << "[Thermal] PTM_SOLVER_EQFIX T_geo-floor release: " << w
                  << " / " << n << " node-updates ("
                  << (100.0 * static_cast<double>(w) / static_cast<double>(n))
                  << "%) would have been raised to T_geo by the removed floor "
                     "and were left at the value the solver computed."
                  << std::endl;
    }
    if (g > 0) {
        std::cerr << "[Thermal] WARNING: the surface-Newton growth-side step "
                     "limiter fired " << g << " time(s). It is supposed to be "
                     "unreachable from the max(T_eq, T1) seed; a below-root "
                     "start caps growth at 1.5x/iteration and can leave the "
                     "surface hundreds of K cold. Investigate before trusting "
                     "surface temperatures." << std::endl;
    }
    if (c > 0) {
        std::cerr << "[Thermal] WARNING: the surface-Newton iteration cap "
                     "(10) was hit " << c << " time(s) without meeting the "
                     "convergence criterion." << std::endl;
    }
}

void surfTemp(Profile& p, double Qs) {
    const double epsilon = p.emissivity;
    const double sigma   = p.config->sigma;
    const double k0      = p.k()[0];
    const double dz0     = p.dz()[0];
    const double T1      = p.T()[1];

    // The whole algorithm -- seed, relaxation, exit criterion, ulp floor and
    // step limiter -- lives in solveSurfaceBC() (thermal_model.h) so that the
    // CPU scalar path, the CPU SIMD batch path and the OpenCL kernel cannot
    // drift apart again. See that header block for the derivation.
    const SurfaceBCSolution s =
        solveSurfaceBC(Qs, T1, k0, dz0, epsilon, sigma, surfaceBCTolK());
    noteSurfaceBCSolution(s);
    p.T()[0] = s.T;
}

// ---------------------------------------------------------------------------
// OUTPUT-EPOCH SURFACE TEMPERATURE (fix/frame-forcing-epoch)
//
// The surface node is MASSLESS: solveSurfaceBC() determines T[0] algebraically
// from (Qs, T[1], k0, dz0). It is therefore not a state variable that is
// carried across a step -- it is a DIAGNOSTIC of whatever epoch it was last
// evaluated at. So a frame filed under output index t must report the surface
// evaluated with the flux AT times[t] against the interior AT times[t]. That
// is one of two independent properties of a frame; the other, that the
// interior is the state at times[t], is what fix/output-time-alignment (PR #66)
// established.
//
// Before this function existed every path reported, instead, whatever the last
// substep it happened to take had left in T[0], whose flux epoch trails
// times[t] by:
//     CPU scalar        one substep      (surfFlux() at the substep START)
//     CPU SIMD batch    half a substep   (flux sampled at the substep MIDPOINT)
//     GPU persistent    a full OUTPUT interval  (one flux value is held across
//                       the whole interval and the frame is stored before it)
// Scored against a 32x-output-cadence reference run on the 80 N tutorial
// crater at 118 output steps, the mean error over sunrise frames was 10.4 K,
// 6.3 K and 188.7 K respectively.
//
// NON-MUTATING, deliberately. Writing the answer back into T[0] would change
// k(T[0]) at the next substep on the GPU and CPU-batch paths -- both refresh
// material properties BEFORE the surface solve -- and so would move the
// trajectory. Then chunk-vs-single-shot equivalence and the byte-identity
// baselines would both be in play for what is an OUTPUT defect. The
// integration is left exactly as it was; only the reported number changes.
//
// The floor is the same one Profile::update_T applies to the column it stores,
// so the reported number cannot sit below a floor the integrator enforces.
double outputSurfaceTemp(const Profile& p, double Qs, double Qb) {
    const double sigma = p.config ? p.config->sigma : ptm::kStefanBoltzmann;
    const SurfaceBCSolution s = solveSurfaceBC(Qs, p.T()[1], p.k()[0], p.dz()[0],
                                               p.emissivity, sigma, surfaceBCTolK());
    // A solve that gives up must be visible in the per-run certificate, the
    // same as every other solve on this path.
    noteSurfaceBCSolution(s);

    const double T_geo = (Qb > 0.0)
        ? std::sqrt(std::sqrt(Qb / (p.emissivity * sigma))) : 0.0;
    const double T_lo  = solverEqFixEnabled() ? ABSOLUTE_ZERO_K
                                              : std::max(ABSOLUTE_ZERO_K, T_geo);
    return std::max(T_lo, std::min(MAX_PHYSICAL_TEMP_K, s.T));
}

double outputSurfaceTempBatch(const ThermalBatchSoA& batch, int face_idx, double Qs) {
    const int f = face_idx;
    const double T0 = batch.T[0 * THERMAL_BATCH_SIZE + f];
    const double T1 = batch.T[1 * THERMAL_BATCH_SIZE + f];
    const double rho0 = batch.rho[0 * THERMAL_BATCH_SIZE + f];

    // k0 exactly as updateMaterialPropertiesBatchSIMD() would set it at the
    // top of the next substep: from the column's CURRENT T[0]. Reading
    // batch.k instead would use the value refreshed one substep ago. This runs
    // once per face per OUTPUT step, not in an inner loop, so it can use the
    // reference dispatch form that test_conductivity_model.cpp pins rather
    // than a fourth hand-copy of the two laws.
    const double k0 = conductivityK(batch.cond_model, T0,
                                    batch.kc[0 * THERMAL_BATCH_SIZE + f],
                                    rho0, batch.R350_val);

    const SurfaceBCSolution s = solveSurfaceBC(Qs, T1, k0, batch.dz[0],
                                               batch.emissivity[f], batch.sigma_sb,
                                               surfaceBCTolK());
    noteSurfaceBCSolution(s);

    // Same T_geo floor advanceTimestep() applies to the column it leaves
    // behind, and -- also matching that path -- NO upper clamp. The scalar
    // flavor above does apply MAX_PHYSICAL_TEMP_K because Profile::update_T
    // does; each helper mirrors the bounds of the path it reports for, rather
    // than inventing a common set. This path refuses to run under
    // PTM_SOLVER_EQFIX, so there is no floor-release branch here.
    const double qb = batch.Qb[f];
    if (qb > 0.0) {
        const double T_geo = std::sqrt(std::sqrt(qb / (batch.emissivity[f] * batch.sigma_sb)));
        if (s.T < T_geo) return T_geo;
    }
    return s.T;
}

void botTemp(Profile& p, double Qb) {
    // Set bottom temperature using heat flux boundary condition
    int n = p.T().size() - 1;
    double kb = p.k()[n];
    double dzb = p.dz()[n-1];
    
    p.T()[n] = p.T()[n-1] + Qb * dzb / kb;
}

// getTimeStep with pow2 optimization
double getTimeStep(const Profile& p, const Planet& planet, double day,
                   const Config& config, bool verbose) {
    double max_fourier = 0.49;
    double dt_min = day;
    int critical_layer = -1;
    double critical_dz = 0.0;
    double critical_kappa = 0.0;

    const auto& z = p.z();
    const auto& kk = p.kappa();
    bool have_kappa = (kk.size() == p.k().size());

    for (size_t i = 1; i < z.size(); i++) {
        double kappa_val;
        if (have_kappa) {
            kappa_val = kk[i];
        } else {
            kappa_val = p.k()[i] / (p.rho()[i] * p.cp()[i]); // fallback
        }
        double dz = (i < z.size() - 1) ? (z[i+1] - z[i]) : (z[i] - z[i-1]);
        double dz2 = dz * dz;
        double dt_stable = max_fourier * dz2 / kappa_val;
        if (dt_stable < dt_min) {
            dt_min = dt_stable;
            critical_layer = static_cast<int>(i);
            critical_dz = dz;
            critical_kappa = kappa_val;
        }
    }

    double dt_day = day / config.NPERDAY;
    dt_min = std::min(dt_min, dt_day);

    if (verbose) {
        std::cout << "Time step calculation:\n"
                  << "  Critical layer: " << critical_layer << "\n"
                  << "  Critical dz: " << critical_dz << " m\n"
                  << "  Critical kappa: " << critical_kappa << " m^2/s\n"
                  << "  Von Neumann dt: " << dt_min << " s\n"
                  << "  Time steps per day: " << (day / dt_min) << std::endl;
    }
    return dt_min;
}

// Model implementation
Model::Model(const Planet& planet, double lat,
             const std::vector<double>& output_times,
             const Eigen::MatrixXd& flux_arr,
             const Config& config,
             bool show_progress)
: planet(planet), lat(lat), flux_arr(flux_arr),
  profile(planet, lat, config, show_progress),
  config(&config), show_progress(show_progress),
  use_full_time_series(false), equilibration_steps(0),
  main_simulation_steps(0)
{
    // Initialize model
    Qs = 0.0;
    t = 0.0;
    N_z = profile.z().size();

    // Initialize flux index
    flux_idx = 1;
    #if defined(__SSE__)
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    #endif

    // Build output_time_points
    if (output_times.empty()) {
        int steps_per_day = std::max(1, config.NPERDAY);
        output_time_points.reserve(steps_per_day);
        for (int i = 0; i < steps_per_day; i++) {
            output_time_points.push_back((planet.day * i) / static_cast<double>(steps_per_day));
        }
    } else {
        output_time_points = output_times;
    }

    // Use output_time_points consistently
    if (!output_time_points.empty()) {
        endtime = output_time_points.back();
    } else {
        endtime = 0.0;
    }
    N_steps = static_cast<int>(output_time_points.size());
    
    // Extract flux arrays from the input matrix if provided (robust to 2xN or Nx2)
    if (flux_arr.size() > 0) {
        std::vector<std::pair<double,double>> tv;
        tv.reserve(static_cast<size_t>(flux_arr.rows() * flux_arr.cols()));
        const int R = static_cast<int>(flux_arr.rows());
        const int C = static_cast<int>(flux_arr.cols());
        if (R == 2 && C >= 1) {
            for (int j = 0; j < C; ++j) {
                double tt = flux_arr(0, j);
                double vv = flux_arr(1, j);
                if (std::isfinite(tt) && std::isfinite(vv)) tv.emplace_back(tt, vv);
            }
        } else if (C == 2 && R >= 1) {
            for (int i = 0; i < R; ++i) {
                double tt = flux_arr(i, 0);
                double vv = flux_arr(i, 1);
                if (std::isfinite(tt) && std::isfinite(vv)) tv.emplace_back(tt, vv);
            }
        } else {
            // Unexpected shape; leave arrays empty => fallback to analytic flux
            tv.clear();
        }

        // Sort by time and dedupe identical time stamps
        if (!tv.empty()) {
            std::sort(tv.begin(), tv.end(), [](const auto& a, const auto& b){ return a.first < b.first; });
            flux_times.clear(); flux_values.clear();
            flux_times.reserve(tv.size());
            flux_values.reserve(tv.size());
            double last_t = std::numeric_limits<double>::quiet_NaN();
            for (const auto& p : tv) {
                if (!flux_times.empty() && p.first == last_t) {
                    // overwrite last sample on duplicate time
                    flux_values.back() = p.second;
                } else {
                    flux_times.push_back(p.first);
                    flux_values.push_back(p.second);
                    last_t = p.first;
                }
            }
            // Ensure rolling index valid
            flux_idx = (flux_times.size() > 1 ? 1u : 0u);
        }
        
        // sanitize flux series (strictly increasing, dedupe)
        if (!flux_times.empty()) {
            // Pair and sort if needed
            bool sorted = std::is_sorted(flux_times.begin(), flux_times.end());
            if (!sorted) {
                std::vector<std::pair<double,double>> tv;
                tv.reserve(flux_times.size());
                for (size_t i=0;i<flux_times.size();++i) tv.emplace_back(flux_times[i], flux_values[i]);
                std::stable_sort(tv.begin(), tv.end(),
                                 [](const auto& a, const auto& b){ return a.first < b.first; });
                flux_times.resize(tv.size());
                flux_values.resize(tv.size());
                for (size_t i=0;i<tv.size();++i) { flux_times[i]=tv[i].first; flux_values[i]=tv[i].second; }
            }
            // Deduplicate non-increasing timestamps
            std::vector<double> times_clean;
            std::vector<double> values_clean;
            times_clean.reserve(flux_times.size());
            values_clean.reserve(flux_values.size());
            const double eps = 1e-9;
            for (size_t i=0;i<flux_times.size();++i) {
                if (times_clean.empty() || flux_times[i] > times_clean.back() + eps) {
                    times_clean.push_back(flux_times[i]);
                    values_clean.push_back(flux_values[i]);
                } else {
                    // Keep last value on duplicate timestamps
                    values_clean.back() = flux_values[i];
                }
            }
            flux_times.swap(times_clean);
            flux_values.swap(values_clean);
            // Reset rolling index
            flux_idx = 1;
        }
    }
    
    // ---- Shared spin-up seed (thermal/eq_seed.h) ---------------------------
    // Profile::init_T ran in the initializer list above and produced the
    // legacy flux-blind, latitude-blind T_avg = T_ss/sqrt(2) column. Now that
    // this face's OWN forcing has been extracted, replace it with the shared
    // per-face conductive steady profile. Legacy mode leaves init_T's answer
    // in place, so a `spinup_seed: legacy` run reproduces the old baseline.
    if (config.spinup_seed == EqSeedMode::Physical && !flux_values.empty()) {
        double sum = 0.0;
        for (double v : flux_values) sum += v;
        seed_mean_flux = sum / static_cast<double>(flux_values.size());

        // Lit-face anchor: this path already holds the face's own forcing
        // series, so it gets the same anchor the batch paths get. The rFFT
        // needs an EVENLY SPACED time base; this path's `times` come straight
        // from the caller and are not guaranteed uniform (the duplicate-
        // timestamp cleanup above can leave gaps), so non-uniformity is
        // detected and DECLARED rather than silently spectrum-analyzed.
        EqSeedForcing fc;
        bool have_forcing = false;
        if (flux_times.size() == flux_values.size() && flux_times.size() >= 2) {
            const double dt0 = flux_times[1] - flux_times[0];
            bool uniform = (dt0 > 0.0);
            for (size_t i = 2; i < flux_times.size() && uniform; ++i) {
                const double dti = flux_times[i] - flux_times[i - 1];
                if (std::fabs(dti - dt0) > 1e-6 * dt0) uniform = false;
            }
            fc.flux               = flux_values.data();
            fc.n_samples          = static_cast<int>(flux_values.size());
            fc.span_s             = dt0 * fc.n_samples;
            fc.cycle_s            = planet.day;
            fc.cp0                = planet.cp0;
            fc.cp_coeff           = planet.cpCoeff.empty() ? nullptr
                                                           : planet.cpCoeff.data();
            fc.n_cp_coeff         = static_cast<int>(planet.cpCoeff.size());
            fc.radiation_coupled  = config.seed_forcing_radiation_coupled;
            fc.nonuniform_samples = !uniform;
            fc.refuse_two_timescale = config.seed_anchor_refuse_two_timescale;
            have_forcing = true;
        }
        seed_result    = profile.seedFromMeanFlux(
            seed_mean_flux, planet.Qb,
            have_forcing ? &fc : nullptr, config.seed_anchor);
        seed_applied   = true;
    }

    // Calculate physics-based time step using von Neumann stability criterion
    dt = getTimeStep(profile, planet, planet.day, config, show_progress);
    // Initialize adaptive state
    dt_scale = 1.0;
    last_surface_T = std::numeric_limits<double>::quiet_NaN();
    
    if (show_progress) {
        std::cout << "Model initialized with " << N_z << " layers" << std::endl;
        std::cout << "Time step: " << dt << " s" << std::endl;
        std::cout << "End time: " << endtime << " s" << std::endl;
        std::cout << "Output steps: " << N_steps << std::endl;
    }
}

// Select an adaptive dt using near-surface stability/accuracy and remaining time
double Model::chooseAdaptiveDt(double target_time) {
    // Base constraints
    double to_target = std::max(0.0, target_time - t);
    if (to_target <= 0.0)
        return 0.0;
    // Enforce minimum temporal sampling of diurnal cycle
    const double dt_day = planet.day / std::max(1, config->NPERDAY);

    // Surface-controlled stability bound (Von Neumann-like using top cell)
    double dt_surf = dt_day; // fallback
    if (!profile.dz().empty() && !profile.kappa().empty()) {
        const double dz0    = profile.dz().front();
        const double kappa0 = profile.kappa().front();
        if (kappa0 > 0.0) {
            dt_surf = std::min(dt_surf, config->F * dz0 * dz0 / kappa0);
        }
    }

    // Apply adaptive scaling and cap
    double dt_try = dt_surf * std::min(config->MAX_DT_SCALE, std::max(0.1, dt_scale));

    // Do not step beyond target time
    dt_try = std::min(dt_try, to_target);

    // Event-aware adjustments based on flux time series
    if (config->EVENT_AWARE_DT && flux_times.size() >= 2 && flux_values.size() == flux_times.size()) {
        // Derivative-based limiter
        const double dSdt = std::abs(estimateFluxDerivative(t));
        if (dSdt > config->FLUX_DERIV_THRESH) {
            double dt_event = std::max(config->MIN_EVENT_DT, config->FLUX_DQ_TARGET / dSdt);
            // If near zero-crossing in flux, tighten further
            double S_now = interpolateFlux(t);
            if (std::abs(S_now) < 2.0 * config->FLUX_DQ_TARGET) {
                dt_event = std::max(config->MIN_EVENT_DT, 0.5 * dt_event);
            }
            dt_try = std::min(dt_try, dt_event);
        }

        // Knot-aware limiter: do not cross a large jump between adjacent samples
        auto hi_it = std::lower_bound(flux_times.begin(), flux_times.end(), t);
        size_t hi = (hi_it == flux_times.end()) ? flux_times.size()-1
                                                : std::distance(flux_times.begin(), hi_it);
        if (hi == 0) hi = 1;
        if (hi < flux_times.size()) {
            size_t lo = hi - 1;
            double t0 = flux_times[lo], t1 = flux_times[hi];
            double v0 = flux_values[lo], v1 = flux_values[hi];
            double seg_dt = std::max(1e-9, t1 - t0);
            double dQ = std::abs(v1 - v0);

            // If the segment jump is large, snap to the next knot
            if (dQ >= config->FLUX_DQ_TARGET) {
                double to_knot = std::max(0.0, t1 - t);
                dt_try = std::min(dt_try, std::max(config->MIN_EVENT_DT, to_knot));
            }
        }
    }

    return std::max(1e-6, dt_try);
}

// Clamp a base dt using event-awareness and target_time guard
double Model::eventLimitedDt(double base_dt, double target_time) {
    if (!config || !config->EVENT_AWARE_DT) return base_dt;
    if (flux_times.size() < 2 || flux_values.size() != flux_times.size()) return base_dt;

    double dt_out = base_dt;

    // Derivative-based limiter
    const double dSdt = std::abs(estimateFluxDerivative(t));
    if (dSdt > config->FLUX_DERIV_THRESH) {
        double dt_event = std::max(config->MIN_EVENT_DT, config->FLUX_DQ_TARGET / dSdt);
        double S_now = interpolateFlux(t);
        if (std::abs(S_now) < 2.0 * config->FLUX_DQ_TARGET) {
            dt_event = std::max(config->MIN_EVENT_DT, 0.5 * dt_event);
        }
        dt_out = std::min(dt_out, dt_event);
    }

    // Knot-aware limiter: avoid crossing large jump until we land on the knot
    auto hi_it = std::lower_bound(flux_times.begin(), flux_times.end(), t);
    size_t hi = (hi_it == flux_times.end()) ? flux_times.size()-1
                                            : std::distance(flux_times.begin(), hi_it);
    if (hi == 0) hi = 1;
    if (hi < flux_times.size()) {
        size_t lo = hi - 1;
        double t1 = flux_times[hi];
        double v0 = flux_values[lo], v1 = flux_values[hi];
        double dQ = std::abs(v1 - v0);
        if (dQ >= config->FLUX_DQ_TARGET) {
            double to_knot = std::max(0.0, t1 - t);
            dt_out = std::min(dt_out, std::max(config->MIN_EVENT_DT, to_knot));
        }
    }

    // Do not overshoot the target
    double to_target = std::max(0.0, target_time - t);
    if (to_target > 0.0) dt_out = std::min(dt_out, to_target);

    return std::max(1e-6, dt_out);
}

// Estimate dQ/dt at current time using linear interpolation of flux series
double Model::estimateFluxDerivative(double time) {
    if (flux_times.size() < 2 || flux_values.size() != flux_times.size()) return 0.0;

    const size_t n = flux_times.size();
    // Advance rolling index while time increases; keep within [1, n-1]
    while (flux_idx < n && flux_times[flux_idx] < time) ++flux_idx;
    if (flux_idx == 0)      flux_idx = 1;
    if (flux_idx >= n)      flux_idx = n - 1;

    const size_t hi = flux_idx;
    const size_t lo = hi - 1;
    const double t0 = flux_times[lo];
    const double t1 = flux_times[hi];
    const double dt = (t1 - t0);
    if (dt <= 0.0) return 0.0;
    return (flux_values[hi] - flux_values[lo]) / dt;
}

double topCellFourierNumber(const Profile& profile, double dt) {
    if (profile.dz().empty() || profile.kappa().empty()) return 0.0;
    const double dz0 = profile.dz()[0];
    if (!(dz0 > 0.0) || !(dt > 0.0)) return 0.0;
    return profile.kappa()[0] * dt / (dz0 * dz0);
}

void warnIfTopCellFourierHigh(const Profile& profile, double dt,
                              const char* context, double fo_warn) {
    const double Fo = topCellFourierNumber(profile, dt);
    if (!(Fo > fo_warn)) return;

    // Once per process. An over-refined run substeps tens of thousands of
    // times; a per-step warning would bury the run's real output.
    static std::atomic<bool> warned{false};
    bool expected = false;
    if (!warned.compare_exchange_strong(expected, true)) return;

    // Amplification of the stiffest mode under CN, for scale. |g| < 1 always
    // (A-stable), so this is an accuracy statement, not a stability one.
    const double g = (1.0 - Fo) / (1.0 + Fo);

    // Formatted into a local stream, NOT straight to cerr. Callers upstream
    // leave std::fixed/setprecision(1) sticky on cerr, under which a
    // sub-millimeter dz0 prints as "0.0" -- the one number this message exists
    // to report. (Observed on the m=20 positive control.)
    std::ostringstream msg;
    msg << std::scientific << std::setprecision(4);
    msg << "[Thermal WARNING] Top-cell Fourier number Fo0 = " << Fo
        << " > " << fo_warn << " (" << (context ? context : "?")
        << "): dz0 = " << profile.dz()[0] << " m, dt = " << dt
        << " s. Crank-Nicolson stays stable, but the stiffest mode's "
           "per-step amplification is " << g
        << " (negative = sign-alternating ringing after a sharp "
           "sunrise/shadow transition, damped by |g| per step).\n"
        << "                  Fo0 = pi*m^2/(steps per rotation), so this "
           "grows as the SQUARE of --thermal-m. To refine the grid further, "
           "raise the substep count in proportion to m^2 "
           "(thermal.equilibration_steps_per_day and the main-loop target), "
           "or accept the ringing. The CPU path does NOT substep on Fo "
           "automatically; the GPU persistent kernel does.\n";
    std::cerr << msg.str();
}

std::vector<double> generateAdaptiveGrid(const Planet& planet, const Config& config, bool verbose) {
    // Angular frequency of diurnal forcing
    const double omega = (planet.day > 0.0) ? (2.0 * M_PI / planet.day) : 0.0;

    // Reference time step from diurnal sampling
    const int steps_per_day = std::max(1, config.NPERDAY);
    const double dt_ref = planet.day / static_cast<double>(steps_per_day);

    // Stability threshold for explicit criterion used as accuracy proxy
    const double Fmax = std::min(0.49, config.F);

    // Local property models vs. depth (same as Profile ctor uses)
    auto k_at = [&](double z)->double {
        const double df = std::exp(-z / std::max(1e-12, planet.H));
        return planet.kd + (planet.ks - planet.kd) * df + 0.0; // base lattice conduction kc
    };
    auto rho_at = [&](double z)->double {
        const double df = std::exp(-z / std::max(1e-12, planet.H));
        return planet.rhod + (planet.rhos - planet.rhod) * df;
    };
    auto cp_at = [&](double /*z*/)->double {
        // Use representative cp at T=200K instead of cp0 (which may be negative for polynomial form)
        double cp_representative = planet.cp0;
        double T_representative = 200.0;
        double T_power = T_representative;
        for (size_t j = 0; j < planet.cpCoeff.size(); ++j) {
            cp_representative += planet.cpCoeff[j] * T_power;
            T_power *= T_representative;
        }
        // Ensure cp is positive and reasonable
        if (cp_representative < 100.0) cp_representative = 600.0;  // Fallback to typical lunar regolith value
        return cp_representative;
    };
    auto kappa_at = [&](double z)->double {
        const double k  = std::max(1e-16, k_at(z));
        const double rr = std::max(1e-6,  rho_at(z));
        const double cc = std::max(1e-6,  cp_at(z));
        return k / (rr * cc);
    };

    // Surface skin depth for diurnal
    const double kappa0 = kappa_at(0.0);
    const double delta0 = (omega > 0.0) ? std::sqrt(2.0 * kappa0 / omega) : 0.1; // safe fallback
    const double zmax   = std::max(10.0 * 1e-6, config.b * delta0);               // total model depth

    // Resolution target: layers per skin depth decays with depth
    const int    N_top = std::max(4, config.m);
    const double e_decay = std::max(0.5, config.RES_DECAY_SKINS); // e-folds of δ0
    auto target_layers_per_skin = [&](double z)->double {
        // Depth expressed in surface skin depths
        double s = z / std::max(1e-12, delta0);
        double N = static_cast<double>(N_top) * std::exp(-s / e_decay);
        return std::max<double>(config.MIN_LAYERS_DEEP_PER_SKIN, N);
    };

    // Initialize grid
    std::vector<double> z;
    z.reserve(static_cast<size_t>(config.m * config.b) + 16);
    z.push_back(0.0);

    // Pick initial dz constrained by both resolution near surface and stability proxy
    const double dz0_res  = delta0 / target_layers_per_skin(0.0);
    const double dz0_stab = std::sqrt(std::max(1e-20, kappa0 * dt_ref / Fmax));
    double dz_prev = std::max(1e-6, std::min(dz0_res, dz0_stab));

    // Geometric stretching constraints
    const double Rmax = std::max(1.05, config.R_MAX); // ensure progress > 1

    // Safety guard to avoid runaway layer counts
    const size_t MAX_LAYERS = 100000; // hard cap

    if (verbose) {
        std::cout << "Adaptive grid build:\n"
                  << "  delta0: " << delta0 << " m, zmax: " << zmax << " m\n"
                  << "  dt_ref: " << dt_ref << " s, Fmax: " << Fmax << "\n"
                  << "  dz0_res: " << dz0_res << " m, dz0_stab: " << dz0_stab << " m\n"
                  << "  Rmax: " << Rmax << ", N_top: " << N_top
                  << ", N_deep_min: " << config.MIN_LAYERS_DEEP_PER_SKIN
                  << ", RES_DECAY_SKINS: " << e_decay << std::endl;
    }

    // March downward
    while (z.back() < zmax && z.size() < MAX_LAYERS) {
        const double z_mid   = z.back() + 0.5 * dz_prev; // estimate properties at layer midpoint
        const double kap_mid = kappa_at(z_mid);
        const double delta_m = (omega > 0.0) ? std::sqrt(2.0 * kap_mid / omega) : delta0;

        // Resolution ceiling: desired dz for current depth based on layers per skin depth
        const double Nps     = target_layers_per_skin(z_mid);
        const double dz_res  = std::max(1e-8, delta_m / Nps);

        // Stability/accuracy ceiling: proxy based on explicit criterion with dt_ref
        const double dz_stab = std::sqrt(std::max(1e-20, kap_mid * dt_ref / Fmax));

        // Candidate next thickness (respect growth cap)
        double dz_next = std::min({ Rmax * dz_prev, dz_res, dz_stab });

        // Ensure we always make progress and respect a tiny floor
        const double dz_floor = 1e-8;
        dz_next = std::max(dz_next, std::min(0.5 * dz_prev, dz_floor)); // avoid zero step

        // If the next step would overshoot zmax too much, clamp last layer to land exactly on zmax
        double z_next = z.back() + dz_next;
        if (z_next > zmax) {
            dz_next = zmax - z.back();
            z_next  = zmax;
        }

        // Guard against non-progress due to extreme constraints
        if (dz_next <= 0.0 || !(z_next > z.back())) {
            // Force a minimal increase
            dz_next = std::max(dz_floor, 1.01 * dz_prev);
            z_next = std::min(zmax, z.back() + dz_next);
        }

        z.push_back(z_next);
        dz_prev = dz_next;
    }

    if (z.back() < zmax) {
        // Enforce final depth if loop stopped on MAX_LAYERS
        z.back() = zmax;
    }

    if (verbose) {
        std::cout << "Adaptive grid layers: " << z.size()
                  << " total depth: " << z.back() << " m\n";
    }
    return z;
}

// Warm start (BUGS.md PTM-76). See the declaration in thermal_model.h.
bool Model::restoreState(const float* T_column, int n) {
    if (!T_column || n <= 0 || n != N_z ||
        static_cast<int>(profile.T().size()) != n) {
        return false;
    }
    auto& Tv = profile.T();
    for (int i = 0; i < n; ++i) {
        // float32 -> float64 is exact; the quantization happened at the write.
        Tv[i] = static_cast<double>(T_column[i]);
    }
    // Same order the constructor uses (update_k, update_cp, recomputeKappa), so
    // a restored profile is indistinguishable from a freshly seeded one apart
    // from the temperatures themselves.
    profile.updateThermalProperties(/*update_k_first=*/true);
    warm_started = true;
    return true;
}

bool Model::captureState(float* T_column, int n) const {
    if (!T_column || n <= 0 || n != N_z ||
        static_cast<int>(profile.T().size()) != n) {
        return false;
    }
    const auto& Tv = profile.T();
    for (int i = 0; i < n; ++i) T_column[i] = static_cast<float>(Tv[i]);
    return true;
}

// Run the model
void Model::run() {
    try {
        // Guard: no outputs requested
        if (output_time_points.empty()) {
            if (show_progress) std::cerr << "Model::run: no output time points\n";
            return;
        }

        // DEBUG: Print diagnostic info for first equatorial face (|lat| < threshold)
        // Enabled via config->THERMAL_DEBUG flag (set by --thermal-debug CLI option)
        int my_face_id = debug_face_counter.fetch_add(1);
        double lat_deg = std::abs(lat * 180.0 / M_PI);
        // Only debug the first face that is near the equator
        static std::atomic<bool> equatorial_face_found{false};
        bool is_debug_face = false;
        const bool thermal_debug_enabled = config && config->THERMAL_DEBUG;
        const double debug_lat_threshold = config ? config->DEBUG_LAT_THRESHOLD : 10.0;

        if (thermal_debug_enabled && lat_deg < debug_lat_threshold && !equatorial_face_found.exchange(true)) {
            is_debug_face = true;
        }

        // Construct debug output path
        std::string debug_output_path = "/tmp/thermal_debug.txt";
        if (config && !config->DEBUG_OUTPUT_DIR.empty()) {
            debug_output_path = config->DEBUG_OUTPUT_DIR + "/thermal_debug.txt";
        }

        if (is_debug_face) {
            std::ofstream dbg(debug_output_path);
            dbg << std::setprecision(10);
            dbg << "=== THERMAL MODEL DEBUG OUTPUT ===" << std::endl;
            dbg << "Planet: " << planet.name << std::endl;
            dbg << "Latitude: " << (lat * 180.0 / M_PI) << " deg" << std::endl;
            dbg << "N_z (depth layers): " << N_z << std::endl;
            dbg << "dt: " << dt << " s" << std::endl;
            dbg << "Output time points: " << output_time_points.size() << std::endl;
            dbg << std::endl;

            dbg << "=== PLANET PROPERTIES ===" << std::endl;
            dbg << "day: " << planet.day << " s" << std::endl;
            dbg << "year: " << planet.year << " s" << std::endl;
            dbg << "S (solar constant): " << planet.S << " W/m^2" << std::endl;
            dbg << "albedo: " << planet.albedo << std::endl;
            dbg << "emissivity: " << planet.emissivity << std::endl;
            dbg << "ks (surface k): " << planet.ks << " W/m/K" << std::endl;
            dbg << "kd (deep k): " << planet.kd << " W/m/K" << std::endl;
            dbg << "rhos (surface rho): " << planet.rhos << " kg/m^3" << std::endl;
            dbg << "rhod (deep rho): " << planet.rhod << " kg/m^3" << std::endl;
            dbg << "H (e-folding): " << planet.H << " m" << std::endl;
            dbg << "cp0: " << planet.cp0 << " J/kg/K" << std::endl;
            dbg << "cpCoeff size: " << planet.cpCoeff.size() << std::endl;
            for (size_t j = 0; j < planet.cpCoeff.size(); ++j) {
                dbg << "  cpCoeff[" << j << "] = " << planet.cpCoeff[j] << std::endl;
            }
            dbg << "Qb (internal heat): " << planet.Qb << " W/m^2" << std::endl;
            dbg << std::endl;

            dbg << "=== CONFIG ===" << std::endl;
            dbg << "R350_val: " << profile.R350_val << std::endl;
            dbg << "chi: " << (config ? config->chi : 0.0) << std::endl;
            dbg << std::endl;

            dbg << "=== GRID DEPTHS (first 10 layers) ===" << std::endl;
            for (int j = 0; j < std::min(10, N_z); ++j) {
                dbg << "z[" << j << "] = " << profile.z()[j] << " m";
                if (j > 0) dbg << ", dz = " << profile.dz()[j-1] << " m";
                dbg << std::endl;
            }
            dbg << "Total depth: " << profile.z()[N_z-1] << " m" << std::endl;
            dbg << std::endl;

            dbg << "=== INITIAL PROFILE (depth 0-9) ===" << std::endl;
            for (int j = 0; j < std::min(10, N_z); ++j) {
                dbg << "z[" << j << "]: depth=" << profile.z()[j] << " m"
                    << ", T=" << profile.T()[j] << " K"
                    << ", k=" << profile.k()[j] << " W/m/K"
                    << ", kc=" << profile.kc()[j] << " W/m/K"
                    << ", rho=" << profile.rho()[j] << " kg/m^3"
                    << ", cp=" << profile.cp()[j] << " J/kg/K"
                    << ", kappa=" << profile.kappa()[j] << " m^2/s"
                    << std::endl;
            }
            dbg << std::endl;
            dbg.close();
        }

        // Allocate output arrays BEFORE first write
        T.resize(static_cast<int>(output_time_points.size()), N_z);
        lt.resize(static_cast<int>(output_time_points.size()));

        t = output_time_points[0];
        ProgressBar progressBar(output_time_points.size(), "Thermal Simulation", 50, show_progress);

        // Analytic clear-sky pre-conditioner. This is a FORCING, not a seed:
        // it integrates `equil_days` diurnal cycles of an idealized
        // cos(lat)*cos(h) insolation with no topography, no shadowing, no IR
        // and no indirect flux, and it OVERWRITES the physically correct
        // per-face initial condition the constructor just computed. Off by
        // default (spinup_seed: physical); see thermal/eq_seed.h.
        const int equil_days = static_cast<int>(std::llround(
            analyticSeedCycles(config->spinup_seed, config->analytic_seed_cycles,
                               planet.day, planet.year, config->NYEARSEQ)));
        if (equil_days > 0 && warm_started) {
            // A restored checkpoint is a continuation. Spinning it up again
            // under idealized clear-sky forcing would discard exactly the state
            // the previous chunk spent its integration producing. Skipped, and
            // said out loud -- once per process, because this runs per face.
            static std::atomic<bool> announced{false};
            if (!announced.exchange(true)) {
                std::cerr << "[warm-start] thermal.initial_state_file is set, so the "
                          << equil_days << "-cycle analytic clear-sky pre-conditioner "
                             "is SKIPPED on the CPU scalar path: the restored T(z) is "
                             "already spun up and the pre-conditioner would overwrite "
                             "it." << std::endl;
            }
        } else if (equil_days > 0) {
            fastEquilibrate(equil_days, t);
        }

        // Initialize flux at the first time point
        surfFlux();

        // Store the first output BEFORE the settling loop below.
        //
        // OUTPUT TIME ALIGNMENT. T(0, :) is the state AT output_time_points[0],
        // and zero time has elapsed there, so it is the initial condition
        // exactly -- the same thing the SIMD batch driver writes at its index 0
        // (ptm_thermal_sim.cpp, runThermalModelsForAllFacesBatchSoA). Storing
        // it after the settling loop (which this did until
        // fix/output-time-alignment) filed a column that had already relaxed
        // 0.4*dt toward equilibrium with Qs(t_0) under index 0. Because the
        // surface node's radiative relaxation time is far shorter than dt, that
        // is not a small perturbation: measured 141.5 K max / 41.2 K mean
        // against the shared spin-up seed on tutorial04_crater80N_radoff with
        // equilibration_orbits = 0.
        //
        // It only ever reached a product when there was no equilibration prefix
        // to discard it (equilibration_orbits = 0); with a prefix, index 0 of
        // this matrix is an equilibration step that never gets exported.
        for (int j = 0; j < N_z; j++) T(0, j) = profile.T()[j];
        // ROW 0 IS THE INITIAL CONDITION, VERBATIM -- INCLUDING ITS SURFACE
        // NODE. Deliberately NOT re-solved at times[0], unlike every row below
        // (fix/frame-forcing-epoch). Two shipped gates assert this convention
        // -- check_output_time_alignment.py's seed and checkpoint anchors, and
        // the warm-start G1 gate -- and it is a convention question, not a
        // measurement question, so it is not settled here.
        //
        // What it costs, measured on the 80 N crater: with
        // equilibration_orbits = 0 the analytic seed's surface differs from
        // the surface that balances flux[0] against the seed's own interior by
        // up to 226.93 K on 11,641 of 12,800 faces; on a mid-lunation
        // warm-start seam the restored checkpoint's surface layer differs by
        // 0.0145 K max. A seam landing on a sunrise would be the former, not
        // the latter.
        //
        // Note this row is only EXPORTED as output index 0 when there is no
        // equilibration prefix. With a prefix (the default) row 0 is an
        // equilibration step that is never exported, and the exported index 0
        // is a row below -- which does get the corrected surface.
        lt(0) = output_time_points[0];
        progressBar.update(1);

        // Small initial settling. Unchanged, so every output from index 1 on is
        // bit-identical to before: it perturbs the initial condition by 0.4*dt
        // of integration at Qs(t_0), which the solver then forgets.
        double initial_dt = dt / 10.0;
        for (int i = 0; i < 4; i++) {
            profile.update_T(initial_dt, Qs, planet.Qb);
            t += initial_dt;
        }
        t = output_time_points[0];

        // Remaining outputs
        int num_outputs = static_cast<int>(output_time_points.size());
        // Debug: sample at 24 local times (every 4 timesteps for 96 total = 24 samples)
        std::vector<int> debug_steps;
        if (is_debug_face) {
            int step_interval = std::max(1, num_outputs / 24);
            for (int s = 0; s < num_outputs; s += step_interval) {
                debug_steps.push_back(s);
            }
            // Also make sure we get the last timestep
            if (debug_steps.back() != num_outputs - 1) {
                debug_steps.push_back(num_outputs - 1);
            }
        }

        // Large-dt CN substepping for the main output loop.
        // When config->MAIN_LOOP_STEPS_PER_DAY > 0, divide each output interval
        // into n_sub CN steps sized for ~thermal-diffusion accuracy at the
        // finest depth layer (see note in thermal_model.h). For Moon that
        // gives ~11 substeps per 72-step-per-lunar-day output interval, vs
        // ~69 substeps under the Fourier-stable choice — ~6x fewer
        // profile.update_T calls per face with max |dT| < 10 K on the
        // Moon-benchmark test config. Falls back to the adaptive/Fourier-stable
        // driver when the flag is 0 so existing users who tuned dt_scale or
        // MULTIRATE_ADAPTIVE_STEP get identical behavior.
        const bool use_cn_substep = config && config->MAIN_LOOP_STEPS_PER_DAY > 0;
        const double cn_target_dt = use_cn_substep
            ? (planet.day / static_cast<double>(config->MAIN_LOOP_STEPS_PER_DAY))
            : 0.0;

        for (int i = 1; i < num_outputs; i++) {
            double target_time = output_time_points[i];

            if (use_cn_substep) {
                const double dt_output = target_time - t;
                const int n_sub = std::max(1,
                    static_cast<int>(std::ceil(dt_output / cn_target_dt)));
                const double dt_sub = dt_output / n_sub;
                for (int s = 0; s < n_sub; ++s) {
                    updateOrbit();
                    surfFlux();
                    profile.update_T(dt_sub, Qs, planet.Qb);
                    t += dt_sub;
                }
            } else {
                while (t < target_time) {
                    updateOrbit();
                    surfFlux();

                    double dt_step = config->MULTIRATE_ADAPTIVE_STEP
                                     ? chooseAdaptiveDt(target_time)
                                     : std::min(dt, target_time - t);

                    double T0_before = profile.T()[0];
                    profile.update_T(dt_step, Qs, planet.Qb);
                    double dT0 = std::abs(profile.T()[0] - T0_before);

                    t += dt_step;

                    if (config->MULTIRATE_ADAPTIVE_STEP) {
                        if (!std::isfinite(dT0)) {
                            dt_scale = std::max(0.1, dt_scale * 0.25);
                        } else if (dT0 < 0.25 * config->SURF_DMAX) {
                            dt_scale = std::min(config->MAX_DT_SCALE, dt_scale * config->DT_GROW);
                        } else if (dT0 > config->SURF_DMAX) {
                            dt_scale = std::max(0.1, dt_scale * config->DT_SHRINK);
                        }
                    }
                }
            }

            t = target_time;
            for (int j = 0; j < N_z; j++) T(i, j) = profile.T()[j];
            // FORCING EPOCH (fix/frame-forcing-epoch). The interior is now the
            // state at target_time, but T[0] is still the surface the LAST
            // SUBSTEP solved -- flux at target_time - dt_sub, against the
            // interior at target_time - dt_sub. Re-evaluate the massless
            // surface at this epoch. surfFlux() reads the member `t`, which
            // was just pinned to target_time, so Qs is the flux at times[i].
            //
            // This does not perturb the run: the next substep calls
            // surfFlux() at exactly this t and update_T() then solves the
            // surface from the same (Qs, T[1], k[0]), so the number computed
            // here is the number that solve produces. Nothing is written back
            // into the column -- see outputSurfaceTemp().
            surfFlux();
            T(i, 0) = outputSurfaceTemp(profile, Qs, planet.Qb);
            lt(i) = target_time;
            progressBar.update(i + 1);

            // DEBUG: Print at sample timesteps (24 local times)
            if (is_debug_face && std::find(debug_steps.begin(), debug_steps.end(), i) != debug_steps.end()) {
                std::ofstream dbg(debug_output_path, std::ios::app);
                dbg << std::setprecision(6) << std::fixed;

                // Calculate local time (0-24 hours within the lunar day)
                double time_in_day = std::fmod(t, planet.day);
                double local_hour = 24.0 * time_in_day / planet.day;

                dbg << "=== TIMESTEP " << i << " / " << num_outputs
                    << " | Local Hour: " << local_hour << " ===" << std::endl;
                dbg << "t = " << t << " s (" << (t / planet.day) << " days)" << std::endl;
                dbg << "Qs (surface flux) = " << Qs << " W/m^2" << std::endl;
                dbg << std::endl;

                // Print header
                dbg << std::setw(4) << "idx" << " "
                    << std::setw(10) << "depth(m)" << " "
                    << std::setw(10) << "T(K)" << " "
                    << std::setw(12) << "k(W/m/K)" << " "
                    << std::setw(12) << "kc(W/m/K)" << " "
                    << std::setw(10) << "rho(kg/m3)" << " "
                    << std::setw(10) << "cp(J/kg/K)" << " "
                    << std::setw(12) << "kappa(m2/s)"
                    << std::endl;
                dbg << std::string(90, '-') << std::endl;

                // Print FULL profile (all layers)
                for (int j = 0; j < N_z; ++j) {
                    dbg << std::setw(4) << j << " "
                        << std::setw(10) << profile.z()[j] << " "
                        << std::setw(10) << profile.T()[j] << " "
                        << std::setw(12) << std::scientific << profile.k()[j] << " "
                        << std::setw(12) << profile.kc()[j] << " "
                        << std::fixed << std::setw(10) << profile.rho()[j] << " "
                        << std::setw(10) << profile.cp()[j] << " "
                        << std::scientific << std::setw(12) << profile.kappa()[j]
                        << std::endl;
                }
                dbg << std::endl;
                dbg.close();
            }
        }
        progressBar.finish();

        // DEBUG: Final summary
        if (is_debug_face) {
            std::ofstream dbg(debug_output_path, std::ios::app);
            dbg << std::setprecision(10);
            dbg << "=== FINAL SUMMARY ===" << std::endl;
            dbg << "Min surface T: " << T.col(0).minCoeff() << " K" << std::endl;
            dbg << "Max surface T: " << T.col(0).maxCoeff() << " K" << std::endl;
            dbg << "Mean surface T: " << T.col(0).mean() << " K" << std::endl;
            dbg << std::endl;
            dbg.close();
            std::cerr << "\n*** THERMAL DEBUG: Output written to " << debug_output_path << " ***\n" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "ERROR in thermal model: " << e.what() << std::endl;
    }
}

void Model::updateOrbit() {
    // Calculate solar angle based on time of day
    // FIXED: Add M_PI to make t=0 correspond to midnight
    double hour_angle = 2.0 * M_PI * std::fmod(t, planet.day) / planet.day + M_PI;
    
    // Normalize to 0-2Ï€ range
    hour_angle = std::fmod(hour_angle, 2.0 * M_PI);
    
    // Correct solar zenith angle calculation for equator
    double cos_zenith = std::cos(lat) * std::cos(hour_angle);
    
    // Only apply flux when sun is above horizon
    double flux_factor = std::max(0.0, cos_zenith);

    // Absorptivity (1 - A). Default path uses the constant Bond albedo. With the
    // Hapke directional-hemispheric gate on, A varies with incidence via the
    // analytic A_h(mu0). The default branch keeps the exact prior expression so
    // gate-off runs are byte-identical.
    double one_minus_A = 1.0 - planet.albedo;
    if (planet.use_hapke_albedo && flux_factor > 0.0) {
        double incidence_angle = std::acos(std::min(1.0, flux_factor));
        double A_h = planet.getAlbedoWithIncidence(0.0, lat, incidence_angle);
        one_minus_A = 1.0 - A_h;
    }

    // Calculate absorbed solar flux
    Sabs = planet.S * flux_factor * one_minus_A;
}

// Set surface flux Qs based on time series or default solar calculation
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
    // Determine the next output target (if any)
    double target_time = t + dt;
    if (!output_time_points.empty()) {
        auto it = std::upper_bound(output_time_points.begin(), output_time_points.end(), t);
        if (it != output_time_points.end()) {
            target_time = std::min(target_time, *it);
        }
    }

    // Use adaptive dt that honors flux events and knots
    double actual_dt = chooseAdaptiveDt(target_time);
    if (actual_dt <= 0.0) return;

    // Update orbital position and surface flux at current time
    updateOrbit();
    surfFlux();

    // Snapshot old temperatures
    std::vector<double> old_temps(profile.T().begin(), profile.T().end());

    // Integrate
    profile.update_T(actual_dt, Qs, planet.Qb);

    // Simple safety: if surface changed too much, retry with smaller step once
    double max_surf_change = std::abs(profile.T()[0] - old_temps[0]);
    if (max_surf_change > 5.0) {
        double smaller_dt = std::max(1e-6, 0.5 * actual_dt);
        // Revert safely into aligned_vector
        profile.T().assign(old_temps.begin(), old_temps.end());  // FIX: replace operator=
        profile.update_T(smaller_dt, Qs, planet.Qb);
        actual_dt = smaller_dt;
    }

    // Advance time
    t += actual_dt;
}

// Equilibration is unconditionally stable under Crank-Nicolson (Profile::update_T),
// so the Fourier-stable config dt (~5000 steps/day for Moon) is wildly conservative.
// Accuracy is dominated by thermal-diffusion resolution at the finest depth layer;
// src/cpp/tools/test_equilibration_dt.cpp shows planet.day / 192 gives <1 K worst-case
// error at 24 lunar days of forcing, ~26x fewer steps than the Fourier-stable choice.
void Model::fastEquilibrate(int days, double target_time) {
    if (show_progress) {
        std::cout << "Running fast equilibration for " << days << " days..." << std::endl;
    }

    ProgressBar eqProgressBar(days, "Fast Equilibration", 50, show_progress);

    // Fixed at 192 steps/day (see accuracy note above). Override via
    // config->EQUILIBRATION_STEPS_PER_DAY if future bodies need finer.
    const int steps_per_day = (config && config->EQUILIBRATION_STEPS_PER_DAY > 0)
                                ? config->EQUILIBRATION_STEPS_PER_DAY
                                : 192;
    const double dt_eq = planet.day / static_cast<double>(steps_per_day);

    for (int day = 0; day < days; day++) {
        eqProgressBar.update(day);
        for (int step = 0; step < steps_per_day; step++) {
            const double day_fraction = static_cast<double>(step) / steps_per_day;
            const double hour_angle = 2.0 * M_PI * day_fraction + M_PI;
            const double cos_zenith = std::cos(lat) * std::cos(hour_angle);
            const double flux_factor = std::max(0.0, cos_zenith);
            // Absorptivity: constant Bond albedo by default; gated Hapke A_h(i)
            // when enabled (gate off -> byte-identical to prior expression).
            double one_minus_A = 1.0 - planet.albedo;
            if (planet.use_hapke_albedo && flux_factor > 0.0) {
                double inc = std::acos(std::min(1.0, flux_factor));
                one_minus_A = 1.0 - planet.getAlbedoWithIncidence(0.0, lat, inc);
            }
            const double local_flux = planet.S * flux_factor * one_minus_A;
            profile.update_T(dt_eq, local_flux, planet.Qb);
        }
    }

    if (show_progress) {
        eqProgressBar.finish();
        std::cout << "Fast equilibration complete" << std::endl;
    }

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
    // Calculate skin depth
    double kappa = planet.ks / (planet.rhos * planet.cp0);
    double skin_depth = skinDepth(planet.day, kappa);
    
    // Optimize grid spacing based on skin depth
    config.m = std::max(10, std::min(20, static_cast<int>(100.0 / skin_depth)));
    
    // Growth factor for geometric progression
    config.n = 5;
    
    // Total depth in skin depths
    // OPTIMIZATION: Replace std::sqrt with faster approximation if needed
    config.b = std::max(10, static_cast<int>(std::sqrt(planet.year / planet.day)));
    
    std::cout << "Optimal grid parameters:" << std::endl;
    std::cout << "  Skin depth layers (m): " << config.m << std::endl;
    std::cout << "  Growth factor (n): " << config.n << std::endl;
    std::cout << "  Total depth in skin depths (b): " << config.b << std::endl;
    std::cout << "  Skin depth: " << skin_depth << " m" << std::endl;
    std::cout << "  Total depth: " << skin_depth * config.b << " m" << std::endl;
}

// Interpolate flux at given time using robust bracket search and linear interp
double Model::interpolateFlux(double time) {
    // If no external series, caller should have computed Sabs; be defensive
    if (flux_times.size() == 0 || flux_values.size() != flux_times.size()) {
        return Sabs;
    }

    // Clamp queries outside data range
    if (time <= flux_times.front()) {
        return flux_values.front();
    }
    if (time >= flux_times.back()) {
        return flux_values.back();
    }

    // Find upper bracket hi where flux_times[hi] >= time
    auto it_hi = std::lower_bound(flux_times.begin(), flux_times.end(), time);
    size_t hi = std::distance(flux_times.begin(), it_hi);
    if (hi == 0) hi = 1;                // safety
    if (hi >= flux_times.size()) hi = flux_times.size() - 1;
    size_t lo = hi - 1;

    // Skip any degenerate segment with identical timestamps
    double t0 = flux_times[lo], t1 = flux_times[hi];
    double v0 = flux_values[lo], v1 = flux_values[hi];
    if (t1 <= t0) {
        return v1; // degenerate; return upper value
    }

    // Linear interpolation
    double w = (time - t0) / (t1 - t0);
    double val = v0 + w * (v1 - v0);

    return val;
}

// =============================================================================
// SIMD Batch Thermal Solver Implementation (SIMD-over-Pixels)
// =============================================================================

void ThermalBatchSoA::initialize(int layers, int faces, const Planet& planet, const Config& config) {
    n_layers = std::min(layers, MAX_THERMAL_LAYERS);
    n_faces = std::min(faces, THERMAL_BATCH_SIZE);

    // Store material parameters
    cp0 = planet.cp0;
    R350_val = config.getR350();
    cond_model = config.conductivity_model;
    rho_s = planet.rhos;
    rho_d = planet.rhod;
    k_s = planet.ks;
    k_d = planet.kd;
    H_val = planet.H;
    sigma_sb = SIGMA_SB;

    // Copy cp polynomial coefficients
    n_cp_coeffs = static_cast<int>(planet.cpCoeff.size());
    for (int i = 0; i < std::min(n_cp_coeffs, 8); ++i) {
        cp_coeffs[i] = planet.cpCoeff[i];
    }

    // Initialize surface properties
    for (int f = 0; f < THERMAL_BATCH_SIZE; ++f) {
        emissivity[f] = planet.emissivity;
        flux[f] = 0.0;
        Qb[f] = planet.Qb;
        T_eq_init[f] = 200.0;  // Default initial temperature
    }

    // Zero out all arrays
    std::memset(T, 0, sizeof(T));
    std::memset(cp, 0, sizeof(cp));
    std::memset(k, 0, sizeof(k));
    std::memset(kappa, 0, sizeof(kappa));
    std::memset(rho, 0, sizeof(rho));
    std::memset(kc, 0, sizeof(kc));
    std::memset(a_coef, 0, sizeof(a_coef));
    std::memset(b_coef, 0, sizeof(b_coef));
    std::memset(c_coef, 0, sizeof(c_coef));
    std::memset(d_coef, 0, sizeof(d_coef));
}

void ThermalBatchSoA::copyGridFromProfile(const Profile& ref_profile,
                                          const double* H_per_face) {
    n_layers = ref_profile.nlayers;

    // Copy grid arrays (shared across all faces in batch)
    for (int i = 0; i < n_layers; ++i) {
        z[i] = ref_profile.z()[i];
    }
    for (int i = 0; i < n_layers - 1; ++i) {
        dz[i] = ref_profile.dz()[i];
    }
    for (int i = 0; i < n_layers - 2; ++i) {
        g1[i] = ref_profile.g1()[i];
        g2[i] = ref_profile.g2()[i];
    }

    // Initialize base conductivity and density for all faces.
    // When H_per_face is supplied, each face gets its own exp(-z/H_f) profile;
    // otherwise the batch's shared H_val drives a single profile shared by all
    // faces in the batch (legacy path).
    if (H_per_face) {
        for (int f = 0; f < n_faces; ++f) {
            const double Hf = (H_per_face[f] > 0.0) ? H_per_face[f] : H_val;
            for (int layer = 0; layer < n_layers; ++layer) {
                const double depth = z[layer];
                const double df    = std::exp(-depth / Hf);
                kc [layer * THERMAL_BATCH_SIZE + f] = k_d  + (k_s  - k_d ) * df;
                rho[layer * THERMAL_BATCH_SIZE + f] = rho_d + (rho_s - rho_d) * df;
            }
        }
    } else {
        for (int layer = 0; layer < n_layers; ++layer) {
            const double depth = z[layer];
            const double df    = std::exp(-depth / H_val);
            const double kc_val  = k_d  + (k_s  - k_d ) * df;
            const double rho_val = rho_d + (rho_s - rho_d) * df;
            for (int f = 0; f < n_faces; ++f) {
                kc [layer * THERMAL_BATCH_SIZE + f] = kc_val;
                rho[layer * THERMAL_BATCH_SIZE + f] = rho_val;
            }
        }
    }

    // Initialize temperature to equilibrium value for all layers and faces
    for (int layer = 0; layer < n_layers; ++layer) {
        for (int f = 0; f < n_faces; ++f) {
            T[layer * THERMAL_BATCH_SIZE + f] = T_eq_init[f];
        }
    }
}

void ThermalBatchSoA::setFlux(const double* flux_values, int num_faces) {
    for (int f = 0; f < std::min(num_faces, THERMAL_BATCH_SIZE); ++f) {
        flux[f] = flux_values[f];
    }
    // Zero-pad remaining slots
    for (int f = num_faces; f < THERMAL_BATCH_SIZE; ++f) {
        flux[f] = 0.0;
    }
}

// SIMD-optimized flux setter for contiguous input (column-major flux layout)
void ThermalBatchSoA::setFluxContiguous(const double* flux_values, int num_faces) {
#if defined(AVX512_ENABLED)
    if (num_faces == THERMAL_BATCH_SIZE) {
        // Full batch: single SIMD load of 8 contiguous doubles
        __m512d flux_vec = _mm512_loadu_pd(flux_values);
        _mm512_store_pd(flux, flux_vec);
        return;
    }
#elif defined(AVX2_ENABLED)
    if (num_faces >= 4) {
        // Load first 4 with AVX2
        __m256d flux_vec0 = _mm256_loadu_pd(flux_values);
        _mm256_store_pd(flux, flux_vec0);
        if (num_faces >= THERMAL_BATCH_SIZE) {
            // Load second 4 with AVX2
            __m256d flux_vec1 = _mm256_loadu_pd(flux_values + 4);
            _mm256_store_pd(flux + 4, flux_vec1);
            return;
        }
        // Handle remaining faces (5-7) with scalar
        for (int f = 4; f < num_faces; ++f) {
            flux[f] = flux_values[f];
        }
        for (int f = num_faces; f < THERMAL_BATCH_SIZE; ++f) {
            flux[f] = 0.0;
        }
        return;
    }
#endif
    // Scalar fallback or partial batch
    std::memcpy(flux, flux_values, num_faces * sizeof(double));
    // Zero-pad remaining slots
    for (int f = num_faces; f < THERMAL_BATCH_SIZE; ++f) {
        flux[f] = 0.0;
    }
}

void ThermalBatchSoA::advanceTimestep(double dt) {
    // PTM_SOLVER_EQFIX refusal, NARROWED 2026-08-11.
    //
    // The original refusal rested on three things: this path's private
    // 5-iteration Newton, the missing R1 deviation form, and the T_geo floor.
    // The first is gone -- applySurfaceBCBatch now calls the shared
    // solveSurfaceBC, so this path runs exactly the specification the scalar
    // path does. What remains is the floor: the loop at the end of this
    // function applies it UNCONDITIONALLY and there is no code here that
    // honors the gate's request to release it.
    //
    // So the refusal stays, for one reason and stated as that one reason: a
    // run that sets PTM_SOLVER_EQFIX and gets the floor anyway would be a
    // silent substitution, which is the failure mode the gate exists to
    // prevent. Refuse rather than accept-and-ignore. BUGS.md PTM-216.
    static const bool refuse = solverEqFixEnabled();
    if (refuse) {
        std::cerr << "\n[Thermal] FATAL: PTM_SOLVER_EQFIX asks for the T_geo floor "
                     "to be released, but the CPU SIMD batch path "
                     "(ThermalBatchSoA::advanceTimestep) applies it "
                     "unconditionally.\n"
                     "  Running here under the gate would keep the floor while "
                     "the log claimed it was released, and nothing would say so.\n"
                     "  (The surface-Newton criterion is NOT the issue: this path "
                     "shares solveSurfaceBC with the scalar path as of "
                     "fix/surface-bc-unification.)\n"
                     "  Only the CPU scalar path (Profile::update_T) releases the "
                     "floor. Unset PTM_SOLVER_EQFIX, or use a configuration that "
                     "takes the scalar path." << std::endl;
        std::exit(1);
    }
    updateMaterialPropertiesBatchSIMD(*this);
    applySurfaceBCBatch(*this, dt);
    buildTridiagonalSystemBatch(*this, dt);
    solveThomasBatchSIMD(*this);
    applyBottomBCBatch(*this);

    // Geothermal-equilibrium floor (mirrors the GPU persistent kernel and the
    // scalar Profile::update_T path). In steady darkness the surface radiates
    // the interior heat flux Qb, so no node can sustain a temperature below
    // T_geo = (Qb/(eps*sigma))^0.25. The under-resolved CN substep at a sharp
    // sunlit->shadow transition lets the upper column of a low-inertia face
    // drift numerically below this physical floor; clamping each face's column
    // at T_geo corrects only that sub-floor drift (real PSR floors converge to
    // T_geo from above -> no-op). Per-face, physics-set, NOT a fixed clamp.
    //
    // NOT touched by PTM_SOLVER_EQFIX: this path REFUSES to run under the gate
    // (see advanceTimestep) rather than accept the flag and keep the floor
    // anyway. BUGS.md PTM-216.
    for (int f = 0; f < n_faces; ++f) {
        const double qb = Qb[f];
        if (qb <= 0.0) continue;
        const double T_geo = std::sqrt(std::sqrt(qb / (emissivity[f] * sigma_sb)));
        if (T_geo <= ABSOLUTE_ZERO_K) continue;
        for (int layer = 0; layer < n_layers; ++layer) {
            double& Tlf = T[layer * THERMAL_BATCH_SIZE + f];
            if (Tlf < T_geo) Tlf = T_geo;
        }
    }
}

// Initialize batch grid from z-grid vector
void initializeBatchGrid(ThermalBatchSoA& batch, const std::vector<double>& z_grid,
                         const Planet& planet) {
    batch.n_layers = std::min(static_cast<int>(z_grid.size()), MAX_THERMAL_LAYERS);

    // Copy z values
    for (int i = 0; i < batch.n_layers; ++i) {
        batch.z[i] = z_grid[i];
    }

    // Compute dz. No minimum-thickness clamp, for the reasons derived at the
    // matching site in Profile's constructor: a floor applied to dz alone
    // desynchronizes the stencil's column from the z-indexed material model
    // and from the GPU, and the constraint that actually binds (CN's
    // oscillatory error) is the dimensionless kappa*dt/dz0^2, not a length.
    for (int i = 0; i < batch.n_layers - 1; ++i) {
        batch.dz[i] = z_grid[i+1] - z_grid[i];
    }

    // Compute grid coefficients g1, g2
    for (int i = 0; i < batch.n_layers - 2; ++i) {
        double d3z_val = batch.dz[i+1] * batch.dz[i] * (batch.dz[i+1] + batch.dz[i]);
        batch.g1[i] = 2.0 * batch.dz[i+1] / d3z_val;
        batch.g2[i] = 2.0 * batch.dz[i] / d3z_val;
    }

    // Initialize base conductivity and density for all faces
    for (int layer = 0; layer < batch.n_layers; ++layer) {
        double depth = batch.z[layer];
        double depth_factor = std::exp(-depth / batch.H_val);
        double kc_val = batch.k_d + (batch.k_s - batch.k_d) * depth_factor;
        double rho_val = batch.rho_d + (batch.rho_s - batch.rho_d) * depth_factor;

        for (int f = 0; f < batch.n_faces; ++f) {
            batch.kc[layer * THERMAL_BATCH_SIZE + f] = kc_val;
            batch.rho[layer * THERMAL_BATCH_SIZE + f] = rho_val;
        }
    }
}

// Update material properties (k, cp, kappa) for all faces in batch using SIMD
void updateMaterialPropertiesBatchSIMD(ThermalBatchSoA& batch) {
    const int n = batch.n_layers;
    const double R350 = batch.R350_val;
    const double cp0_val = batch.cp0;
    const int n_coeffs = batch.n_cp_coeffs;
    const double* coeffs = batch.cp_coeffs;

    // Martinez & Siegler (2021): scalar over the whole batch, taken before the
    // SIMD blocks so those stay bit-for-bit unchanged for the default law.
    if (batch.cond_model == ConductivityModel::Martinez2021) {
        for (int layer = 0; layer < n; ++layer) {
            double* T_ptr     = batch.layer_ptr(batch.T, layer);
            double* k_ptr     = batch.layer_ptr(batch.k, layer);
            double* cp_ptr    = batch.layer_ptr(batch.cp, layer);
            double* kappa_ptr = batch.layer_ptr(batch.kappa, layer);
            double* rho_ptr   = batch.layer_ptr(batch.rho, layer);
            for (int f = 0; f < THERMAL_BATCH_SIZE; ++f) {
                const double Tf = T_ptr[f];
                k_ptr[f] = martinezK(Tf, rho_ptr[f]);
                double cpv = cp0_val;
                if (n_coeffs > 0) {
                    double result = coeffs[n_coeffs - 1];
                    for (int j = n_coeffs - 2; j >= 0; --j) result = result * Tf + coeffs[j];
                    cpv = cp0_val + Tf * result;
                }
                cp_ptr[f] = cpv;
                kappa_ptr[f] = k_ptr[f] / (rho_ptr[f] * cpv);
            }
        }
        return;
    }

#if defined(AVX512_ENABLED)
    __m512d one = _mm512_set1_pd(1.0);
    __m512d r350_vec = _mm512_set1_pd(R350);
    __m512d cp0_vec = _mm512_set1_pd(cp0_val);

    for (int layer = 0; layer < n; ++layer) {
        double* T_ptr = batch.layer_ptr(batch.T, layer);
        double* kc_ptr = batch.layer_ptr(batch.kc, layer);
        double* k_ptr = batch.layer_ptr(batch.k, layer);
        double* cp_ptr = batch.layer_ptr(batch.cp, layer);
        double* kappa_ptr = batch.layer_ptr(batch.kappa, layer);
        double* rho_ptr = batch.layer_ptr(batch.rho, layer);

        // Load T for 8 faces
        __m512d T_vec = _mm512_load_pd(T_ptr);
        __m512d kc_vec = _mm512_load_pd(kc_ptr);
        __m512d rho_vec = _mm512_load_pd(rho_ptr);

        // Compute k = kc * (1 + R350 * T^3)
        __m512d T2 = _mm512_mul_pd(T_vec, T_vec);
        __m512d T3 = _mm512_mul_pd(T2, T_vec);
        __m512d factor = _mm512_fmadd_pd(r350_vec, T3, one);
        __m512d k_vec = _mm512_mul_pd(kc_vec, factor);
        _mm512_store_pd(k_ptr, k_vec);

        // Compute cp using Horner's method
        __m512d cp_result;
        if (n_coeffs > 0) {
            cp_result = _mm512_set1_pd(coeffs[n_coeffs - 1]);
            for (int j = n_coeffs - 2; j >= 0; --j) {
                __m512d cj = _mm512_set1_pd(coeffs[j]);
                cp_result = _mm512_fmadd_pd(cp_result, T_vec, cj);
            }
            cp_result = _mm512_fmadd_pd(T_vec, cp_result, cp0_vec);
        } else {
            cp_result = cp0_vec;
        }
        _mm512_store_pd(cp_ptr, cp_result);

        // Compute kappa = k / (rho * cp)
        __m512d rho_cp = _mm512_mul_pd(rho_vec, cp_result);
        __m512d kappa_vec = _mm512_div_pd(k_vec, rho_cp);
        _mm512_store_pd(kappa_ptr, kappa_vec);
    }

#elif defined(AVX2_ENABLED)
    __m256d one = _mm256_set1_pd(1.0);
    __m256d r350_vec = _mm256_set1_pd(R350);
    __m256d cp0_vec = _mm256_set1_pd(cp0_val);

    for (int layer = 0; layer < n; ++layer) {
        double* T_ptr = batch.layer_ptr(batch.T, layer);
        double* kc_ptr = batch.layer_ptr(batch.kc, layer);
        double* k_ptr = batch.layer_ptr(batch.k, layer);
        double* cp_ptr = batch.layer_ptr(batch.cp, layer);
        double* kappa_ptr = batch.layer_ptr(batch.kappa, layer);
        double* rho_ptr = batch.layer_ptr(batch.rho, layer);

        // Process 4 faces at a time (two AVX2 loads for 8 faces)
        for (int base = 0; base < THERMAL_BATCH_SIZE; base += 4) {
            __m256d T_vec = _mm256_load_pd(T_ptr + base);
            __m256d kc_vec = _mm256_load_pd(kc_ptr + base);
            __m256d rho_vec = _mm256_load_pd(rho_ptr + base);

            // Compute k = kc * (1 + R350 * T^3)
            __m256d T2 = _mm256_mul_pd(T_vec, T_vec);
            __m256d T3 = _mm256_mul_pd(T2, T_vec);
            __m256d factor = _mm256_fmadd_pd(r350_vec, T3, one);
            __m256d k_vec = _mm256_mul_pd(kc_vec, factor);
            _mm256_store_pd(k_ptr + base, k_vec);

            // Compute cp using Horner's method
            __m256d cp_result;
            if (n_coeffs > 0) {
                cp_result = _mm256_set1_pd(coeffs[n_coeffs - 1]);
                for (int j = n_coeffs - 2; j >= 0; --j) {
                    __m256d cj = _mm256_set1_pd(coeffs[j]);
                    cp_result = _mm256_fmadd_pd(cp_result, T_vec, cj);
                }
                cp_result = _mm256_fmadd_pd(T_vec, cp_result, cp0_vec);
            } else {
                cp_result = cp0_vec;
            }
            _mm256_store_pd(cp_ptr + base, cp_result);

            // Compute kappa = k / (rho * cp)
            __m256d rho_cp = _mm256_mul_pd(rho_vec, cp_result);
            __m256d kappa_vec = _mm256_div_pd(k_vec, rho_cp);
            _mm256_store_pd(kappa_ptr + base, kappa_vec);
        }
    }

#else
    // Scalar fallback
    for (int layer = 0; layer < n; ++layer) {
        for (int f = 0; f < batch.n_faces; ++f) {
            int idx = layer * THERMAL_BATCH_SIZE + f;
            double t = batch.T[idx];

            // Compute k = kc * (1 + R350 * T^3)
            batch.k[idx] = batch.kc[idx] * (1.0 + R350 * t * t * t);

            // Compute cp using Horner's method
            double cp_val = cp0_val;
            if (n_coeffs > 0) {
                double poly = coeffs[n_coeffs - 1];
                for (int j = n_coeffs - 2; j >= 0; --j) {
                    poly = poly * t + coeffs[j];
                }
                cp_val = cp0_val + t * poly;
            }
            batch.cp[idx] = cp_val;

            // Compute kappa = k / (rho * cp)
            batch.kappa[idx] = batch.k[idx] / (batch.rho[idx] * cp_val);
        }
    }
#endif
}

// Apply surface boundary condition using Newton-Raphson iteration
// Surface boundary condition for the SoA batch path.
//
// This used to be a fourth, private implementation of the surface energy
// balance, and it was the worst one in the codebase as well as the default
// mesh solver (thermal.use_gpu defaults false). It seeded from the PREVIOUS
// T[0] -- off-bracket at any transition -- ran exactly 5 iterations with no
// convergence test, had no step limiter, and applied a hard [2.7, 500] K clamp
// INSIDE the Newton loop, which can terminate iteration on a legitimately hot
// face and is exactly the arbitrary cutoff this project has ruled against.
// Measured single-call errors against a float64 Brent root, all one-signed
// warm: +22.958 K (0 -> 127 W/m^2), +13.416 K (0 -> 255), +28.686 K
// (1197 -> 0 at T1 = 100 K) -- i.e. a warm bias at every terminator crossing,
// shadow exit and eclipse transition.
//
// It is deleted. The batch keeps its SoA layout; the per-face solve is the
// shared solveSurfaceBC(). The AVX-512/AVX2 kernels are NOT reinstated in
// vector form: the exit criterion is per-lane data-dependent, so a vector
// version would either run every lane to the worst lane's iteration count or
// need mask bookkeeping -- a fifth implementation to keep in step with the
// other three, which is precisely how this divergence was created. The surface
// solve is one scalar root find per face against a 50-layer Thomas sweep, so
// it is not where this path's time goes.
void applySurfaceBCBatch(ThermalBatchSoA& batch, double dt) {
    (void)dt;   // the surface balance is quasi-static; dt enters the interior solve
    const double sigma  = batch.sigma_sb;
    const double dz0    = batch.dz[0];
    const double tol_K  = surfaceBCTolK();
    const double* k0p   = batch.layer_ptr(batch.k, 0);
    const double* T1p   = batch.layer_ptr(batch.T, 1);
    double*       T0p   = batch.layer_ptr(batch.T, 0);

    for (int f = 0; f < batch.n_faces; ++f) {
        const SurfaceBCSolution s = solveSurfaceBC(
            batch.flux[f], T1p[f], k0p[f], dz0, batch.emissivity[f], sigma, tol_K);
        noteSurfaceBCSolution(s);
        T0p[f] = s.T;
    }
}

// Apply bottom boundary condition (geothermal heat flux)
void applyBottomBCBatch(ThermalBatchSoA& batch) {
    const int last = batch.n_layers - 1;
    const double dz_bottom = batch.dz[last - 1];

#if defined(AVX512_ENABLED)
    __m512d dz_vec = _mm512_set1_pd(dz_bottom);

    __m512d T_prev = _mm512_load_pd(batch.layer_ptr(batch.T, last - 1));
    __m512d k_last = _mm512_load_pd(batch.layer_ptr(batch.k, last));
    __m512d Qb_vec = _mm512_load_pd(batch.Qb);

    // T_bottom = T_prev + Qb * dz / k
    __m512d delta_T = _mm512_div_pd(_mm512_mul_pd(Qb_vec, dz_vec), k_last);
    __m512d T_bottom = _mm512_add_pd(T_prev, delta_T);

    _mm512_store_pd(batch.layer_ptr(batch.T, last), T_bottom);

#elif defined(AVX2_ENABLED)
    __m256d dz_vec = _mm256_set1_pd(dz_bottom);

    for (int base = 0; base < THERMAL_BATCH_SIZE; base += 4) {
        __m256d T_prev = _mm256_load_pd(batch.layer_ptr(batch.T, last - 1) + base);
        __m256d k_last = _mm256_load_pd(batch.layer_ptr(batch.k, last) + base);
        __m256d Qb_vec = _mm256_load_pd(batch.Qb + base);

        __m256d delta_T = _mm256_div_pd(_mm256_mul_pd(Qb_vec, dz_vec), k_last);
        __m256d T_bottom = _mm256_add_pd(T_prev, delta_T);

        _mm256_store_pd(batch.layer_ptr(batch.T, last) + base, T_bottom);
    }

#else
    // Scalar fallback
    for (int f = 0; f < batch.n_faces; ++f) {
        double T_prev = batch.T[(last - 1) * THERMAL_BATCH_SIZE + f];
        double k_val = batch.k[last * THERMAL_BATCH_SIZE + f];
        double Qb = batch.Qb[f];

        batch.T[last * THERMAL_BATCH_SIZE + f] = T_prev + Qb * dz_bottom / k_val;
    }
#endif
}

// Build Crank-Nicolson tridiagonal system for batch
void buildTridiagonalSystemBatch(ThermalBatchSoA& batch, double dt) {
    const int n = batch.n_layers;
    const int n_int = n - 2;  // Interior nodes (excluding surface and bottom)

    if (n_int <= 0) return;

#if defined(AVX512_ENABLED)
    __m512d half = _mm512_set1_pd(0.5);
    __m512d one = _mm512_set1_pd(1.0);
    __m512d dt_vec = _mm512_set1_pd(dt);

    for (int j = 0; j < n_int; ++j) {
        const int i = j + 1;  // Interior node index [1..n-2]
        const double g1_val = batch.g1[i - 1];
        const double g2_val = batch.g2[i - 1];

        __m512d g1_vec = _mm512_set1_pd(g1_val);
        __m512d g2_vec = _mm512_set1_pd(g2_val);

        // Load data for 8 faces at layers i-1, i, i+1
        __m512d k_im1 = _mm512_load_pd(batch.layer_ptr(batch.k, i - 1));
        __m512d k_i = _mm512_load_pd(batch.layer_ptr(batch.k, i));
        __m512d rho_i = _mm512_load_pd(batch.layer_ptr(batch.rho, i));
        __m512d cp_i = _mm512_load_pd(batch.layer_ptr(batch.cp, i));

        __m512d T_im1 = _mm512_load_pd(batch.layer_ptr(batch.T, i - 1));
        __m512d T_i = _mm512_load_pd(batch.layer_ptr(batch.T, i));
        __m512d T_ip1 = _mm512_load_pd(batch.layer_ptr(batch.T, i + 1));

        // Compute coupling coefficients
        __m512d alpha = _mm512_mul_pd(g1_vec, k_im1);
        __m512d beta = _mm512_mul_pd(g2_vec, k_i);
        __m512d rho_cp = _mm512_mul_pd(rho_i, cp_i);
        __m512d inv_rhocp = _mm512_div_pd(dt_vec, rho_cp);

        // a[j] = -0.5 * inv_rhocp * alpha
        __m512d a_vec = _mm512_mul_pd(_mm512_mul_pd(_mm512_set1_pd(-0.5), inv_rhocp), alpha);

        // b[j] = 1.0 + 0.5 * inv_rhocp * (alpha + beta)
        __m512d alpha_beta = _mm512_add_pd(alpha, beta);
        __m512d b_vec = _mm512_fmadd_pd(_mm512_mul_pd(half, inv_rhocp), alpha_beta, one);

        // c[j] = -0.5 * inv_rhocp * beta
        __m512d c_vec = _mm512_mul_pd(_mm512_mul_pd(_mm512_set1_pd(-0.5), inv_rhocp), beta);

        // d[j] = T_i + 0.5 * inv_rhocp * (alpha*T_{i-1} - (alpha+beta)*T_i + beta*T_{i+1})
        __m512d term1 = _mm512_mul_pd(alpha, T_im1);
        __m512d term2 = _mm512_mul_pd(alpha_beta, T_i);
        __m512d term3 = _mm512_mul_pd(beta, T_ip1);
        __m512d explicit_part = _mm512_add_pd(_mm512_sub_pd(term1, term2), term3);
        __m512d d_vec = _mm512_fmadd_pd(_mm512_mul_pd(half, inv_rhocp), explicit_part, T_i);

        _mm512_store_pd(batch.layer_ptr(batch.a_coef, j), a_vec);
        _mm512_store_pd(batch.layer_ptr(batch.b_coef, j), b_vec);
        _mm512_store_pd(batch.layer_ptr(batch.c_coef, j), c_vec);
        _mm512_store_pd(batch.layer_ptr(batch.d_coef, j), d_vec);
    }

    // Fold boundary values into RHS and zero boundary coefficients
    // d[0] -= a[0] * T[0]
    __m512d T_surf = _mm512_load_pd(batch.layer_ptr(batch.T, 0));
    __m512d a_0 = _mm512_load_pd(batch.layer_ptr(batch.a_coef, 0));
    __m512d d_0 = _mm512_load_pd(batch.layer_ptr(batch.d_coef, 0));
    d_0 = _mm512_fnmadd_pd(a_0, T_surf, d_0);
    _mm512_store_pd(batch.layer_ptr(batch.d_coef, 0), d_0);
    _mm512_store_pd(batch.layer_ptr(batch.a_coef, 0), _mm512_setzero_pd());

    // d[n_int-1] -= c[n_int-1] * T[n-1]
    __m512d T_bottom = _mm512_load_pd(batch.layer_ptr(batch.T, n - 1));
    __m512d c_last = _mm512_load_pd(batch.layer_ptr(batch.c_coef, n_int - 1));
    __m512d d_last = _mm512_load_pd(batch.layer_ptr(batch.d_coef, n_int - 1));
    d_last = _mm512_fnmadd_pd(c_last, T_bottom, d_last);
    _mm512_store_pd(batch.layer_ptr(batch.d_coef, n_int - 1), d_last);
    _mm512_store_pd(batch.layer_ptr(batch.c_coef, n_int - 1), _mm512_setzero_pd());

#elif defined(AVX2_ENABLED)
    __m256d half = _mm256_set1_pd(0.5);
    __m256d one = _mm256_set1_pd(1.0);
    __m256d dt_vec = _mm256_set1_pd(dt);

    for (int j = 0; j < n_int; ++j) {
        const int i = j + 1;
        const double g1_val = batch.g1[i - 1];
        const double g2_val = batch.g2[i - 1];

        __m256d g1_vec = _mm256_set1_pd(g1_val);
        __m256d g2_vec = _mm256_set1_pd(g2_val);

        for (int base = 0; base < THERMAL_BATCH_SIZE; base += 4) {
            __m256d k_im1 = _mm256_load_pd(batch.layer_ptr(batch.k, i - 1) + base);
            __m256d k_i = _mm256_load_pd(batch.layer_ptr(batch.k, i) + base);
            __m256d rho_i = _mm256_load_pd(batch.layer_ptr(batch.rho, i) + base);
            __m256d cp_i = _mm256_load_pd(batch.layer_ptr(batch.cp, i) + base);

            __m256d T_im1 = _mm256_load_pd(batch.layer_ptr(batch.T, i - 1) + base);
            __m256d T_i = _mm256_load_pd(batch.layer_ptr(batch.T, i) + base);
            __m256d T_ip1 = _mm256_load_pd(batch.layer_ptr(batch.T, i + 1) + base);

            __m256d alpha = _mm256_mul_pd(g1_vec, k_im1);
            __m256d beta = _mm256_mul_pd(g2_vec, k_i);
            __m256d rho_cp = _mm256_mul_pd(rho_i, cp_i);
            __m256d inv_rhocp = _mm256_div_pd(dt_vec, rho_cp);

            __m256d a_vec = _mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(-0.5), inv_rhocp), alpha);
            __m256d alpha_beta = _mm256_add_pd(alpha, beta);
            __m256d b_vec = _mm256_fmadd_pd(_mm256_mul_pd(half, inv_rhocp), alpha_beta, one);
            __m256d c_vec = _mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(-0.5), inv_rhocp), beta);

            __m256d term1 = _mm256_mul_pd(alpha, T_im1);
            __m256d term2 = _mm256_mul_pd(alpha_beta, T_i);
            __m256d term3 = _mm256_mul_pd(beta, T_ip1);
            __m256d explicit_part = _mm256_add_pd(_mm256_sub_pd(term1, term2), term3);
            __m256d d_vec = _mm256_fmadd_pd(_mm256_mul_pd(half, inv_rhocp), explicit_part, T_i);

            _mm256_store_pd(batch.layer_ptr(batch.a_coef, j) + base, a_vec);
            _mm256_store_pd(batch.layer_ptr(batch.b_coef, j) + base, b_vec);
            _mm256_store_pd(batch.layer_ptr(batch.c_coef, j) + base, c_vec);
            _mm256_store_pd(batch.layer_ptr(batch.d_coef, j) + base, d_vec);
        }
    }

    // Fold boundary values (AVX2)
    for (int base = 0; base < THERMAL_BATCH_SIZE; base += 4) {
        __m256d T_surf = _mm256_load_pd(batch.layer_ptr(batch.T, 0) + base);
        __m256d a_0 = _mm256_load_pd(batch.layer_ptr(batch.a_coef, 0) + base);
        __m256d d_0 = _mm256_load_pd(batch.layer_ptr(batch.d_coef, 0) + base);
        d_0 = _mm256_fnmadd_pd(a_0, T_surf, d_0);
        _mm256_store_pd(batch.layer_ptr(batch.d_coef, 0) + base, d_0);
        _mm256_store_pd(batch.layer_ptr(batch.a_coef, 0) + base, _mm256_setzero_pd());

        __m256d T_bottom = _mm256_load_pd(batch.layer_ptr(batch.T, n - 1) + base);
        __m256d c_last = _mm256_load_pd(batch.layer_ptr(batch.c_coef, n_int - 1) + base);
        __m256d d_last = _mm256_load_pd(batch.layer_ptr(batch.d_coef, n_int - 1) + base);
        d_last = _mm256_fnmadd_pd(c_last, T_bottom, d_last);
        _mm256_store_pd(batch.layer_ptr(batch.d_coef, n_int - 1) + base, d_last);
        _mm256_store_pd(batch.layer_ptr(batch.c_coef, n_int - 1) + base, _mm256_setzero_pd());
    }

#else
    // Scalar fallback
    for (int j = 0; j < n_int; ++j) {
        const int i = j + 1;
        const double g1_val = batch.g1[i - 1];
        const double g2_val = batch.g2[i - 1];

        for (int f = 0; f < batch.n_faces; ++f) {
            int idx_im1 = (i - 1) * THERMAL_BATCH_SIZE + f;
            int idx_i = i * THERMAL_BATCH_SIZE + f;
            int idx_ip1 = (i + 1) * THERMAL_BATCH_SIZE + f;

            double alpha = g1_val * batch.k[idx_im1];
            double beta = g2_val * batch.k[idx_i];
            double inv_rhocp = dt / (batch.rho[idx_i] * batch.cp[idx_i]);

            int coef_idx = j * THERMAL_BATCH_SIZE + f;
            batch.a_coef[coef_idx] = -0.5 * inv_rhocp * alpha;
            batch.b_coef[coef_idx] = 1.0 + 0.5 * inv_rhocp * (alpha + beta);
            batch.c_coef[coef_idx] = -0.5 * inv_rhocp * beta;
            batch.d_coef[coef_idx] = batch.T[idx_i] + 0.5 * inv_rhocp *
                (alpha * batch.T[idx_im1] - (alpha + beta) * batch.T[idx_i] + beta * batch.T[idx_ip1]);
        }
    }

    // Fold boundaries (scalar)
    for (int f = 0; f < batch.n_faces; ++f) {
        batch.d_coef[f] -= batch.a_coef[f] * batch.T[f];
        batch.a_coef[f] = 0.0;

        int last_idx = (n_int - 1) * THERMAL_BATCH_SIZE + f;
        batch.d_coef[last_idx] -= batch.c_coef[last_idx] * batch.T[(n - 1) * THERMAL_BATCH_SIZE + f];
        batch.c_coef[last_idx] = 0.0;
    }
#endif
}

// Solve 8 tridiagonal systems in parallel using SIMD Thomas algorithm
void solveThomasBatchSIMD(ThermalBatchSoA& batch) {
    const int n = batch.n_layers - 2;  // Interior nodes only

    if (n <= 0) return;

#if defined(AVX512_ENABLED)
    // Forward sweep: eliminate sub-diagonal
    for (int j = 1; j < n; ++j) {
        __m512d a_j = _mm512_load_pd(batch.layer_ptr(batch.a_coef, j));
        __m512d b_jm1 = _mm512_load_pd(batch.layer_ptr(batch.b_coef, j - 1));
        __m512d c_jm1 = _mm512_load_pd(batch.layer_ptr(batch.c_coef, j - 1));
        __m512d d_jm1 = _mm512_load_pd(batch.layer_ptr(batch.d_coef, j - 1));
        __m512d b_j = _mm512_load_pd(batch.layer_ptr(batch.b_coef, j));
        __m512d d_j = _mm512_load_pd(batch.layer_ptr(batch.d_coef, j));

        // m = a[j] / b[j-1]
        __m512d m = _mm512_div_pd(a_j, b_jm1);

        // b[j] -= m * c[j-1]
        b_j = _mm512_fnmadd_pd(m, c_jm1, b_j);

        // d[j] -= m * d[j-1]
        d_j = _mm512_fnmadd_pd(m, d_jm1, d_j);

        _mm512_store_pd(batch.layer_ptr(batch.b_coef, j), b_j);
        _mm512_store_pd(batch.layer_ptr(batch.d_coef, j), d_j);
    }

    // Back substitution: x[n-1] = d[n-1] / b[n-1]
    {
        int last = n - 1;
        __m512d d_last = _mm512_load_pd(batch.layer_ptr(batch.d_coef, last));
        __m512d b_last = _mm512_load_pd(batch.layer_ptr(batch.b_coef, last));
        __m512d x_last = _mm512_div_pd(d_last, b_last);
        // Store to T[last+1] because coefficients are for interior nodes [1..n_layers-2]
        _mm512_store_pd(batch.layer_ptr(batch.T, last + 1), x_last);
    }

    // x[j] = (d[j] - c[j] * x[j+1]) / b[j]
    for (int j = n - 2; j >= 0; --j) {
        __m512d d_j = _mm512_load_pd(batch.layer_ptr(batch.d_coef, j));
        __m512d c_j = _mm512_load_pd(batch.layer_ptr(batch.c_coef, j));
        __m512d b_j = _mm512_load_pd(batch.layer_ptr(batch.b_coef, j));
        __m512d x_jp1 = _mm512_load_pd(batch.layer_ptr(batch.T, j + 2));

        __m512d numer = _mm512_fnmadd_pd(c_j, x_jp1, d_j);
        __m512d x_j = _mm512_div_pd(numer, b_j);
        _mm512_store_pd(batch.layer_ptr(batch.T, j + 1), x_j);
    }

#elif defined(AVX2_ENABLED)
    // Forward sweep (AVX2: process 4 faces at a time)
    for (int j = 1; j < n; ++j) {
        for (int base = 0; base < THERMAL_BATCH_SIZE; base += 4) {
            __m256d a_j = _mm256_load_pd(batch.layer_ptr(batch.a_coef, j) + base);
            __m256d b_jm1 = _mm256_load_pd(batch.layer_ptr(batch.b_coef, j - 1) + base);
            __m256d c_jm1 = _mm256_load_pd(batch.layer_ptr(batch.c_coef, j - 1) + base);
            __m256d d_jm1 = _mm256_load_pd(batch.layer_ptr(batch.d_coef, j - 1) + base);
            __m256d b_j = _mm256_load_pd(batch.layer_ptr(batch.b_coef, j) + base);
            __m256d d_j = _mm256_load_pd(batch.layer_ptr(batch.d_coef, j) + base);

            __m256d m = _mm256_div_pd(a_j, b_jm1);
            b_j = _mm256_fnmadd_pd(m, c_jm1, b_j);
            d_j = _mm256_fnmadd_pd(m, d_jm1, d_j);

            _mm256_store_pd(batch.layer_ptr(batch.b_coef, j) + base, b_j);
            _mm256_store_pd(batch.layer_ptr(batch.d_coef, j) + base, d_j);
        }
    }

    // Back substitution (AVX2)
    {
        int last = n - 1;
        for (int base = 0; base < THERMAL_BATCH_SIZE; base += 4) {
            __m256d d_last = _mm256_load_pd(batch.layer_ptr(batch.d_coef, last) + base);
            __m256d b_last = _mm256_load_pd(batch.layer_ptr(batch.b_coef, last) + base);
            __m256d x_last = _mm256_div_pd(d_last, b_last);
            _mm256_store_pd(batch.layer_ptr(batch.T, last + 1) + base, x_last);
        }
    }

    for (int j = n - 2; j >= 0; --j) {
        for (int base = 0; base < THERMAL_BATCH_SIZE; base += 4) {
            __m256d d_j = _mm256_load_pd(batch.layer_ptr(batch.d_coef, j) + base);
            __m256d c_j = _mm256_load_pd(batch.layer_ptr(batch.c_coef, j) + base);
            __m256d b_j = _mm256_load_pd(batch.layer_ptr(batch.b_coef, j) + base);
            __m256d x_jp1 = _mm256_load_pd(batch.layer_ptr(batch.T, j + 2) + base);

            __m256d numer = _mm256_fnmadd_pd(c_j, x_jp1, d_j);
            __m256d x_j = _mm256_div_pd(numer, b_j);
            _mm256_store_pd(batch.layer_ptr(batch.T, j + 1) + base, x_j);
        }
    }

#else
    // Scalar fallback: solve each face independently
    for (int f = 0; f < batch.n_faces; ++f) {
        // Forward sweep
        for (int j = 1; j < n; ++j) {
            int idx = j * THERMAL_BATCH_SIZE + f;
            int idx_prev = (j - 1) * THERMAL_BATCH_SIZE + f;

            double m = batch.a_coef[idx] / batch.b_coef[idx_prev];
            batch.b_coef[idx] -= m * batch.c_coef[idx_prev];
            batch.d_coef[idx] -= m * batch.d_coef[idx_prev];
        }

        // Back substitution
        int last_idx = (n - 1) * THERMAL_BATCH_SIZE + f;
        batch.T[(n) * THERMAL_BATCH_SIZE + f] = batch.d_coef[last_idx] / batch.b_coef[last_idx];

        for (int j = n - 2; j >= 0; --j) {
            int idx = j * THERMAL_BATCH_SIZE + f;
            int t_idx_next = (j + 2) * THERMAL_BATCH_SIZE + f;
            int t_idx = (j + 1) * THERMAL_BATCH_SIZE + f;

            batch.T[t_idx] = (batch.d_coef[idx] - batch.c_coef[idx] * batch.T[t_idx_next]) / batch.b_coef[idx];
        }
    }
#endif
}

} // namespace thermal