/*
 * Europa Exosphere Physics Module - macOS Compatible Version
 * Removes unsupported AVX2/AVX512 transcendental functions
 * Fixes Eigen sparse matrix operations
 */

#ifndef EXOSPHERE_PHYSICS_H
#define EXOSPHERE_PHYSICS_H

#include <vector>
#include <array>
#include <cmath>
#include <memory>
#include <limits>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

#include "distance_cache.h"
#include "spherical_index.h"

// Disable problematic AVX features on macOS
#ifdef __APPLE__
    #ifdef AVX2_ENABLED
        #include <immintrin.h>
        // Note: AVX2 transcendental functions not available on macOS
        #define AVX2_BASIC_ONLY
    #endif
    // Disable AVX512 on macOS due to missing intrinsics
    #undef AVX512_ENABLED
#else
    #ifdef AVX2_ENABLED
        #include <immintrin.h>
    #endif
    #ifdef AVX512_ENABLED
        #include <immintrin.h>
    #endif
#endif

namespace ptm {
namespace exosphere {

// Physical constants
namespace constants {
    constexpr double BOLTZMANN_K = 1.380649e-23;          // J/K
    constexpr double WATER_MOLECULAR_MASS = 2.99151e-26;  // kg (H2O)
    constexpr double WATER_MOLAR_MASS = 0.01801528;       // kg/mol
    constexpr double GAS_CONSTANT_R = 8.314462618;        // J/(mol·K)
    constexpr double EUROPA_RADIUS = 1.5608e6;            // m
    constexpr double EUROPA_GRAVITY = 1.314;              // m/s^2
    constexpr double EUROPA_ESCAPE_VELOCITY = 2025.0;     // m/s
    constexpr double ICE_DENSITY = 917.0;                 // kg/m^3
    constexpr double MIN_ICE_THICKNESS = 1e-9;            // 1 nm minimum thickness
}

// Modified Bessel function of the second kind, order zero, K0(x) for x > 0.
// This is the ONLY implementation in the exosphere module; every transport
// kernel (and the CLI in ptm_exosphere_main.cpp) must call this one.
//
// x <= 2: exact power series (A&S 9.6.13 with n = 0),
//     K0(x) = -(ln(x/2) + gamma) I0(x) + sum_{k>=1} H_k (x^2/4)^k / (k!)^2,
//     I0(x) = sum_{k>=0} (x^2/4)^k / (k!)^2,  H_k = 1 + 1/2 + ... + 1/k.
//   With x^2/4 <= 1 the k = 13 term is < 1e-19 relative, so 12 terms are
//   exact to double precision.
// x > 2: A&S 9.8.6 polynomial in 2/x (|error| < 1.9e-7 on sqrt(x) e^x K0).
//
// History: the previous x <= 2 branch dropped the -gamma constant and used a
// series that is not I0(x); it overestimated K0 by +104% at x = 1 and +507%
// just below x = 2, with a 6x jump at x = 2 (EXO-01). test_bessel_k0_and_areas
// pins both branches and the continuity at x = 2.
inline double besselK0(double x) {
    if (x <= 0.0) {
        return std::numeric_limits<double>::infinity();  // log singularity at 0
    }
    if (x <= 2.0) {
        constexpr double EULER_GAMMA = 0.57721566490153286061;
        const double q = 0.25 * x * x;   // (x/2)^2, in (0, 1]
        double term = 1.0;               // (x^2/4)^k / (k!)^2
        double harmonic = 0.0;           // H_k
        double i0 = 1.0;
        double tail = 0.0;
        for (int k = 1; k <= 12; ++k) {
            term *= q / (static_cast<double>(k) * static_cast<double>(k));
            harmonic += 1.0 / static_cast<double>(k);
            i0 += term;
            tail += harmonic * term;
        }
        return -(std::log(0.5 * x) + EULER_GAMMA) * i0 + tail;
    } else {
        double y = 2.0 / x;
        double ak = 1.25331414 + y * (-0.07832358 + y * (0.02189568 + y * (-0.01062446 + 
                    y * (0.00587872 + y * (-0.00251540 + y * 0.00053208)))));
        return std::exp(-x) / std::sqrt(x) * ak;
    }
}

// Spatial post-smoothing radius [rad] shared by the deposition field
// (BallisticTransport::computeTransportProduct) and the sublimation and
// sputtering fields (PTMExosphereIntegration::computeDiurnalAverages).
//
// OFF (0, a true no-op) by default (Paul, 2026-09-22). The smoothing existed to
// suppress mesh-topology (pentagon) artifacts. Its default of 20x the mean
// face angular radius never took effect before the EXO-02 area fix (km^2 areas
// made it ~0.008 deg); after that fix it silently became ~8 deg on ico6. The
// EXO-01 K0 fix already removed most of the topology bias it targeted.
// Measured 2026-09-22 (ico6, uniform 130 K, zonal cos^4 source; departure of
// deposition from zonal symmetry): pre-fix kernel rms 1.5e-3; fixed kernel
// with smoothing off rms 3.6e-5, at the estimator floor (2.6e-5); an 8 deg
// top-hat smoothing ADDS rms 4.3e-2 (max 0.14) of its own cap-edge artifact.
//
// Opt in with EXO_DEP_SMOOTH_DEG=<degrees>, 0 < degrees < 30; "0" means off.
// Any other value (unparsable, negative, >= 30) throws instead of being
// silently ignored. The same radius applies to all three fields.
inline double spatialSmoothingRadiusRad() {
    const char* envp = std::getenv("EXO_DEP_SMOOTH_DEG");
    if (envp == nullptr || *envp == '\0') return 0.0;
    char* end = nullptr;
    const double deg = std::strtod(envp, &end);
    if (end == envp || *end != '\0' || !std::isfinite(deg) || deg < 0.0 || deg >= 30.0) {
        throw std::invalid_argument(std::string("EXO_DEP_SMOOTH_DEG='") + envp +
                                    "' is invalid: expected degrees in [0, 30) (0 = off)");
    }
    return deg * M_PI / 180.0;
}

// Vapor pressure physics
class VaporPressure {
    // Lookup table for fast vapor pressure evaluation.
    // Replaces std::pow(10.0, ...) with linear interpolation.
    // Table covers T_MIN to T_MAX K with TABLE_SIZE equally spaced points.
    static constexpr double T_MIN = 10.0;
    static constexpr double T_MAX = 300.0;
    static constexpr int TABLE_SIZE = 29001;  // 0.01 K spacing
    static constexpr double T_STEP = (T_MAX - T_MIN) / (TABLE_SIZE - 1);
    static constexpr double INV_T_STEP = (TABLE_SIZE - 1) / (T_MAX - T_MIN);

    struct LookupTable {
        double values[TABLE_SIZE];
        LookupTable() {
            // ln(10) factor for converting log10 to natural log (faster than pow(10,...))
            constexpr double LN10 = 2.302585092994046;
            for (int i = 0; i < TABLE_SIZE; ++i) {
                double T = T_MIN + i * T_STEP;
                double log10_p = -2663.5 / T + 12.537;
                values[i] = std::exp(LN10 * log10_p);
            }
        }
    };

    static const LookupTable& getTable() {
        static const LookupTable table;
        return table;
    }

public:
    // Fast vapor pressure via lookup table with linear interpolation.
    // Marti & Mauersberger (1993): log10(P[Pa]) = -2663.5/T + 12.537
    // Max interpolation error < 0.001% with 0.01 K spacing.
    static inline double calculateVaporPressure(double temperature_K) {
        if (temperature_K <= 0.0) return 0.0;

        // Clamp to table range; above T_MAX use direct formula (rare)
        if (temperature_K >= T_MAX) {
            double log10_p = -2663.5 / temperature_K + 12.537;
            return std::exp(2.302585092994046 * log10_p);
        }
        if (temperature_K <= T_MIN) {
            return getTable().values[0];
        }

        // Linear interpolation
        double idx_f = (temperature_K - T_MIN) * INV_T_STEP;
        int idx = static_cast<int>(idx_f);
        double frac = idx_f - idx;

        const auto& tbl = getTable().values;
        return tbl[idx] + frac * (tbl[idx + 1] - tbl[idx]);
    }
    
    static inline double calculateVaporDensity(double pressure_Pa, double temperature_K) {
        if (temperature_K <= 0.0 || pressure_Pa <= 0.0) return 0.0;
        return pressure_Pa * constants::WATER_MOLAR_MASS / 
               (constants::GAS_CONSTANT_R * temperature_K);
    }
    
    static inline double calculateThermalVelocity(double temperature_K) {
        if (temperature_K <= 0.0) return 0.0;
        return std::sqrt(8.0 * constants::BOLTZMANN_K * temperature_K / 
                        (M_PI * constants::WATER_MOLECULAR_MASS));
    }
    
    static inline double calculateSublimationFlux(double temperature_K) {
        if (temperature_K <= 0.0) return 0.0;
        
        double vapor_pressure = calculateVaporPressure(temperature_K);
        double vapor_density = calculateVaporDensity(vapor_pressure, temperature_K);
        double thermal_velocity = calculateThermalVelocity(temperature_K);
        
        return vapor_density * thermal_velocity / 4.0;
    }
    
    // Vector operations
    static void calculateVaporPressureVector(const std::vector<double>& temperatures,
                                            std::vector<double>& pressures);
    static void calculateSublimationFluxVector(const std::vector<double>& temperatures,
                                              std::vector<double>& fluxes);
    
    // Simplified AVX2 operations (basic arithmetic only on macOS)
#ifdef AVX2_ENABLED
    static void calculateThermalVelocityVectorAVX2(const double* temperatures,
                                                   double* velocities,
                                                   size_t count);
#endif
};

// Fast acos lookup table for transport kernel evaluation.
// Replaces std::acos() (~10-20 ns) with table lookup + linear interpolation (~2 ns).
// Used in computeTransportProduct() inner loop (~100B+ evaluations per diurnal computation).
class FastAcos {
    static constexpr int TABLE_SIZE = 65536;
    static constexpr double INV_HALF_SIZE = TABLE_SIZE * 0.5;

    struct LookupTable {
        double values[TABLE_SIZE + 1];
        LookupTable() {
            for (int i = 0; i <= TABLE_SIZE; ++i) {
                double x = -1.0 + 2.0 * i / TABLE_SIZE;
                values[i] = std::acos(std::max(-1.0, std::min(1.0, x)));
            }
        }
    };

    static const LookupTable& getTable() {
        static const LookupTable table;
        return table;
    }

public:
    // Fast acos via lookup table with linear interpolation.
    // Input: x in [-1, 1]. Output: acos(x) in [0, pi].
    // Max error: ~1.5e-5 radians with 65536-point table.
    static inline double evaluate(double x) {
        double idx_f = (x + 1.0) * INV_HALF_SIZE;
        int idx = static_cast<int>(idx_f);
        if (idx < 0) return M_PI;
        if (idx >= TABLE_SIZE) return 0.0;
        double frac = idx_f - idx;
        const auto& tbl = getTable().values;
        return tbl[idx] + frac * (tbl[idx + 1] - tbl[idx]);
    }
};

// Per-call options for BallisticTransport::computeTransportProduct (PTM-642).
struct TransportCallOptions {
    // Remove the Jeans-escaping share of each source's Maxwellian. false = the
    // source is already a bound population (its escaping share was removed
    // upstream, e.g. the non-escaping sputtered H2O), so it must not escape twice.
    bool apply_jeans_escape = true;
    // Use a call-local spatial index sized for THIS call's temperatures when the
    // persistent index (built on the first call) is too small, leaving the
    // persistent index untouched so the sublimation transport stays bit-stable.
    bool private_index_if_needed = false;
};

// Global mass totals of one transport call [kg/s]. source = deposited +
// jeans_escaped + unscattered to rounding; unscattered is emission from sources
// the kernel could not launch (T <= 0 or no kernel weight) and should be 0.
struct TransportTotals {
    double source = 0.0;
    double deposited = 0.0;
    double jeans_escaped = 0.0;
    double unscattered = 0.0;
};

// Ballistic transport physics
class BallisticTransport {
private:
    double gravity_;
    double body_radius_;
    double escape_velocity_;

    // Old distance cache (deprecated, kept for compatibility)
    mutable std::vector<std::vector<double>> distance_cache_;
    mutable bool cache_valid_;

    // New optimized distance cache
    mutable DistanceCache optimized_cache_;
    mutable bool optimized_cache_built_;

    // Adaptive rebuild controller
    mutable AdaptiveRebuildController rebuild_controller_;

    // Spatial index for O(N log N) neighbor queries
    mutable SphericalSpatialIndex spatial_index_;
    mutable bool spatial_index_built_;
    // Largest angular query radius [rad] the spatial index was built for;
    // queries beyond it are silently truncated, so smoothing checks it.
    mutable double spatial_index_query_radius_ = 0.0;

    // Sinkhorn scaling factor cache for warm-start optimization
    mutable Eigen::VectorXd sinkhorn_row_scale_cache_;
    mutable Eigen::VectorXd sinkhorn_col_scale_cache_;
    mutable bool sinkhorn_cache_valid_;

    // Skip Sinkhorn area-bias correction (faster build, ~2-5% area bias)
    bool skip_sinkhorn_ = false;

    // Declare mass check used by the implementation
    void verifyMassConservation(const Eigen::SparseMatrix<double>& matrix) const;

public:
    BallisticTransport(double gravity = constants::EUROPA_GRAVITY,
                      double radius = constants::EUROPA_RADIUS,
                      double escape_v = constants::EUROPA_ESCAPE_VELOCITY);
    
    ~BallisticTransport() = default;
    
    // Great circle distance - inline for performance
    inline double greatCircleDistance(const std::array<double, 3>& pos1,
                                     const std::array<double, 3>& pos2) const {
        double r1 = std::sqrt(pos1[0]*pos1[0] + pos1[1]*pos1[1] + pos1[2]*pos1[2]);
        double r2 = std::sqrt(pos2[0]*pos2[0] + pos2[1]*pos2[1] + pos2[2]*pos2[2]);
        if (r1 <= 0.0 || r2 <= 0.0) return 0.0;
        std::array<double, 3> n1 = {pos1[0]/r1, pos1[1]/r1, pos1[2]/r1};
        std::array<double, 3> n2 = {pos2[0]/r2, pos2[1]/r2, pos2[2]/r2};
        double dot = n1[0]*n2[0] + n1[1]*n2[1] + n1[2]*n2[2];
        dot = std::max(-1.0, std::min(1.0, dot));
        double angle = std::acos(dot);
        return body_radius_ * angle;
    }
    
    double calculateEscapeFraction(double temperature_K) const;
    
    inline double transportProbabilityDensity(double distance_m, double temperature_K) const {
        if (distance_m <= 0.0 || temperature_K <= 0.0) return 0.0;
        double sigma2 = constants::BOLTZMANN_K * temperature_K / constants::WATER_MOLECULAR_MASS;
        double sigma4 = sigma2 * sigma2;
        double xi = gravity_ * distance_m / (2.0 * sigma2);
        if (xi > 10.0) return 0.0;
        double prefactor = gravity_ * gravity_ * distance_m / (4.0 * sigma4);
        return prefactor * besselK0(xi);
    }
    
    double transportProbabilityIntegrated(double distance_m, double solid_angle, double temperature_K) const;
    double calculateSolidAngle(double face_area, double distance) const;

    // NEW: declaration to match the .cpp definition
    void buildTransportMatrixOptimized(
        const std::vector<std::array<double, 3>>& face_centers,
        const std::vector<double>& face_areas,
        const std::vector<double>& temperatures,
        Eigen::SparseMatrix<double>& transport_matrix,
        bool use_cache);

    // Build transport matrix (declaration only; implemented in .cpp)
    Eigen::SparseMatrix<double> buildTransportMatrix(
        const std::vector<std::array<double, 3>>& face_centers,
        const std::vector<double>& face_areas,
        const std::vector<double>& temperatures,
        double threshold = 1e-10) const;

    // NEW: Optimized transport matrix build with distance cache
    Eigen::SparseMatrix<double> buildTransportMatrixCached(
        const std::vector<std::array<double, 3>>& face_centers,
        const std::vector<double>& face_areas,
        const std::vector<double>& temperatures,
        double threshold = 1e-10) const;

    // NEW: Check if matrix rebuild is needed (adaptive)
    bool needsMatrixRebuild(const std::vector<double>& temperatures) const {
        return rebuild_controller_.needsRebuild(temperatures);
    }

    // NEW: Force next rebuild
    void forceMatrixRebuild() const { rebuild_controller_.forceRebuild(); }

    // NEW: Get rebuild statistics
    void printRebuildStats() const { rebuild_controller_.printStats(); }

    // NEW: Configure adaptive rebuild thresholds
    void setRebuildThresholds(double relative, double absolute) {
        rebuild_controller_.setRelativeThreshold(relative);
        rebuild_controller_.setAbsoluteThreshold(absolute);
    }

    // NEW: Check if distance cache is built
    bool isDistanceCacheBuilt() const { return optimized_cache_built_ && optimized_cache_.isValid(); }

    // NEW: Get distance cache memory usage
    size_t getDistanceCacheMemory() const { return optimized_cache_.memoryUsage(); }

    // NEW: Spatially-indexed transport matrix build - O(N*k) where k = avg neighbors
    Eigen::SparseMatrix<double> buildTransportMatrixIndexed(
        const std::vector<std::array<double, 3>>& face_centers,
        const std::vector<double>& face_areas,
        const std::vector<double>& temperatures,
        double threshold = 1e-10) const;

    // NEW: Smooth kernel transport matrix - no shell binning artifacts
    // Uses direct kernel evaluation with proper spherical integration
    Eigen::SparseMatrix<double> buildTransportMatrixSmooth(
        const std::vector<std::array<double, 3>>& face_centers,
        const std::vector<double>& face_areas,
        const std::vector<double>& temperatures,
        double threshold = 1e-10) const;

    // Skip Sinkhorn area-bias correction in buildTransportMatrixSmooth
    void setSkipSinkhorn(bool skip) { skip_sinkhorn_ = skip; }

    // Matrix-free transport product: computes deposition = M * emission
    // without forming the full sparse matrix. Uses O(N + N*threads) memory
    // instead of O(nnz) for the full matrix. `options` and `totals` are
    // optional; the defaults reproduce the original behavior exactly.
    void computeTransportProduct(
        const std::vector<std::array<double, 3>>& face_centers,
        const std::vector<double>& face_areas,
        const std::vector<double>& temperatures,
        const std::vector<double>& emission,
        std::vector<double>& deposition,
        double threshold = 1e-10,
        const TransportCallOptions& options = TransportCallOptions(),
        TransportTotals* totals = nullptr) const;

    // Area-weighted smoothing of a per-face scalar over the spatial-index
    // neighborhood within `smooth_radius_rad`. A radius <= 0 is a no-op
    // (output = input); callers pass spatialSmoothingRadiusRad(), which is 0
    // unless EXO_DEP_SMOOTH_DEG opts in. Throws if the radius exceeds the
    // spatial index's query radius. Copies input -> output when the spatial
    // index has not yet been built.
    void smoothField(
        const std::vector<double>& face_areas,
        const std::vector<double>& input,
        std::vector<double>& output,
        double smooth_radius_rad = 0.0) const;

    // NEW: Check if spatial index is built
    bool isSpatialIndexBuilt() const { return spatial_index_built_; }

    // NEW: Get spatial index memory usage
    size_t getSpatialIndexMemory() const { return spatial_index_.memoryUsage(); }

    // NEW: Print spatial index statistics
    void printSpatialIndexStats() const { spatial_index_.printStatistics(); }
}; // <-- ensure class ends with semicolon

// Exosphere state
struct ExosphereState {
    std::vector<double> vapor_pressure;      // Pa
    std::vector<double> vapor_density;       // kg/m³
    std::vector<double> thermal_velocity;    // m/s
    std::vector<double> emission_flux;       // kg/(m²·s)
    std::vector<double> deposition_flux;     // kg/(m²·s)
    std::vector<double> net_flux;           // kg/(m²·s)
    std::vector<double> accumulated_mass;    // kg/m²
    std::vector<double> ice_thickness;      // m
    std::vector<double> escape_flux;        // kg/(m²·s)
    
    explicit ExosphereState(size_t num_faces) {
        vapor_pressure.resize(num_faces, 0.0);
        vapor_density.resize(num_faces, 0.0);
        thermal_velocity.resize(num_faces, 0.0);
        emission_flux.resize(num_faces, 0.0);
        deposition_flux.resize(num_faces, 0.0);
        net_flux.resize(num_faces, 0.0);
        accumulated_mass.resize(num_faces, 0.0);
        ice_thickness.resize(num_faces, 0.0);
        escape_flux.resize(num_faces, 0.0);
    }
    
    void updateIceThickness() {
        for (size_t i = 0; i < ice_thickness.size(); ++i) {
            ice_thickness[i] = accumulated_mass[i] / constants::ICE_DENSITY;
        }
    }
};

// Main exosphere model
class ExosphereModel {
private:
    size_t num_faces_;
    std::shared_ptr<ExosphereState> state_;
    std::unique_ptr<BallisticTransport> transport_;
    
    std::vector<std::array<double, 3>> face_centers_;
    std::vector<double> face_areas_;
    std::vector<double> face_latitudes_;
    std::vector<double> face_longitudes_;
    
    Eigen::SparseMatrix<double> transport_matrix_;
    bool transport_matrix_valid_;
    
    // Solver parameters
    double time_step_;
    double convergence_threshold_;
    int max_iterations_;
    
    // Statistics tracking
    double total_mass_sublimated_;
    double total_mass_deposited_;
    double total_mass_escaped_;
    double initial_equatorial_thickness_;
    double initial_polar_thickness_;

    // Verbosity and optimization control
    bool verbose_ = true;
    bool skip_matrix_rebuild_ = false;

    // Stored temperatures for matrix-free transport in evolveTimeStep()
    std::vector<double> last_temperatures_;

public:
    explicit ExosphereModel(size_t num_faces);
    ~ExosphereModel() = default;
    
    void setInitialIceDistribution(const std::vector<double>& initial_mass);
    void setLatitudinalIceDistribution(double equatorial_thickness = 1e-4, 
                                       double polar_thickness = 1e-3);
    
    void setInitialIceDistribution(const std::string& filename);
    void exportIceDistribution(const std::string& filename) const;

    // Setup methods
    void setMeshGeometry(const std::vector<std::array<double, 3>>& face_centers,
                        const std::vector<double>& face_areas);
    void calculateFaceCoordinates();
    
    // Update methods
    void updateTemperatures(const std::vector<double>& temperatures);
    void updateTransportMatrix(bool force_rebuild = false);
    
    // Time evolution methods
    void evolveTimeStep(double dt);
    void evolveMultipleSteps(double dt, int n_steps);
    
    // Solver methods
    void solveSteadyState(int max_iterations = 1000, double tolerance = 1e-6);
    void solveMatrixEquation(double dt);
    void solveImplicitEuler(double dt);
    
    // Analysis methods
    void calculateMassBalance();
    void calculateLatitudinalDistribution(std::vector<double>& lat_bins,
                                         std::vector<double>& mass_bins,
                                         int n_bins = 18);
    
    // Export methods
    void exportState(const std::string& filename) const;
    void exportTransportMatrix(const std::string& filename) const;
    void exportStatistics(const std::string& filename) const;
    
    // Accessors
    const ExosphereState& getState() const { return *state_; }
    ExosphereState& getMutableState() { return *state_; }
    const Eigen::SparseMatrix<double>& getTransportMatrix() const { return transport_matrix_; }
    
    // Configuration
    void setTimeStep(double dt) { time_step_ = dt; }
    void setConvergenceThreshold(double tol) { convergence_threshold_ = tol; }
    void setMaxIterations(int max_iter) { max_iterations_ = max_iter; }
    
    // Statistics
    double getTotalMassSublimated() const { return total_mass_sublimated_; }
    double getTotalMassDeposited() const { return total_mass_deposited_; }
    double getTotalMassEscaped() const { return total_mass_escaped_; }

    // Verbosity control for suppressing repeated messages
    void setVerbose(bool verbose) { verbose_ = verbose; }
    bool isVerbose() const { return verbose_; }

    // Skip transport matrix rebuild (for diurnal averaging when we only need fluxes)
    void setSkipMatrixRebuild(bool skip) { skip_matrix_rebuild_ = skip; }
    bool isSkipMatrixRebuild() const { return skip_matrix_rebuild_; }

    // Skip Sinkhorn area-bias correction (faster matrix build)
    void setSkipSinkhorn(bool skip) { if (transport_) transport_->setSkipSinkhorn(skip); }

    // Matrix-free transport product: computes deposition from emission
    // without building the full transport matrix. For one-shot SpMV.
    void computeTransportProduct(
        const std::vector<double>& temperatures,
        const std::vector<double>& emission,
        std::vector<double>& deposition,
        const TransportCallOptions& options = TransportCallOptions(),
        TransportTotals* totals = nullptr);

    // Area-weighted smoothing of a per-face scalar field. Forwards to
    // BallisticTransport::smoothField using face_areas_ and the transport's
    // spatial index (radius <= 0 is a no-op). If transport is not
    // initialized, copies input -> output.
    void smoothField(
        const std::vector<double>& input,
        std::vector<double>& output,
        double smooth_radius_rad = 0.0);

    // Optional: Method to check ice availability at each face
    std::vector<bool> getIcePresenceMask() const {
        std::vector<bool> mask(num_faces_);
        for (size_t i = 0; i < num_faces_; ++i) {
            mask[i] = (state_->accumulated_mass[i] > 0.0);
        }
        return mask;
    }
    
    // Optional: Get total ice mass on surface
    double getTotalIceMass() const {
        double total = 0.0;
        for (size_t i = 0; i < num_faces_; ++i) {
            total += state_->accumulated_mass[i] * face_areas_[i];
        }
        return total;
    }
};

// NOTE: Transport now uses physically integrated self-cap (face area) probability
// to compute diagonal entries instead of leftover remainder. This removes
// mesh-topology amplification of anisotropy.
// Ring-density correction: Each source column is debiased by shell area ratio
// to remove icosahedral packing imprint. Disable with EXO_DISABLE_RING_CORR=1.
// Performance notes:
//  - Ring-density correction now uses direct O(N) angular binning (no per-column sort).
//  - EXO_SHELL_WIDTH_FACTOR controls shell angular thickness (default 2.5).
//  - Set EXO_DISABLE_RING_CORR=1 to disable debiasing (for A/B tests).

} // namespace exosphere
} // namespace ptm

#endif // EXOSPHERE_PHYSICS_H