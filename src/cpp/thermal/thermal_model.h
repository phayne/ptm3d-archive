#ifndef THERMAL_MODEL_H
#define THERMAL_MODEL_H

#include <vector>
#include <cmath>
#include <cstdint>
#include <string>
#include <memory>
#include <iostream>  // Add this for std::cout
#include <chrono>    // for ProgressBar elapsed/ETA
#include <sstream>   // for ProgressBar duration formatting
#include <iomanip>   // for ProgressBar duration formatting
#include "eigen_config.h"
#include <map>       // Add this for std::map
#include "planetary_properties.h"  // Add this include
#include "conductivity_model.h"    // Selectable k(T,rho) laws (Hayne 2017 / Martinez 2021)
#include "../maps/spatial_map.h"  // Add this include for SpatialMap class

namespace thermal {

// Fast power computation helpers - inline for header optimization
inline double pow2(double x) { return x * x; }
inline double pow3(double x) { return x * x * x; }
inline double pow4(double x) { double x2 = x * x; return x2 * x2; }

// Simple aligned allocator for vector data buffers
template <typename T, std::size_t Alignment = 64>
struct AlignedAllocator {
    using value_type = T;
    AlignedAllocator() noexcept {}
    template <class U> AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    T* allocate(std::size_t n) {
        void* p = nullptr;
    #if defined(_MSC_VER)
        p = _aligned_malloc(n * sizeof(T), Alignment);
        if (!p) throw std::bad_alloc();
    #else
        if (posix_memalign(&p, Alignment, n * sizeof(T)) != 0) throw std::bad_alloc();
    #endif
        return reinterpret_cast<T*>(p);
    }
    void deallocate(T* p, std::size_t) noexcept {
    #if defined(_MSC_VER)
        _aligned_free(p);
    #else
        free(p);
    #endif
    }
    template <class U> struct rebind { using other = AlignedAllocator<U, Alignment>; };
};

template <typename T>
using aligned_vector = std::vector<T, AlignedAllocator<T, 64>>;

// Forward declarations
class Planet;
class Profile;
class Config;

class ProgressBar {
private:
    int total;
    int current;
    int barWidth;
    std::string prefix;
    bool enabled;
    bool completed;
    int last_percentage;
    std::chrono::steady_clock::time_point start_time;
    bool started;

    // Format a duration in seconds as "Xs", "Xm Ys", or "XhYm" depending
    // on magnitude. Keep it short so the progress line stays readable.
    static std::string format_duration(double seconds) {
        if (seconds < 0.0 || !std::isfinite(seconds)) return "?";
        if (seconds < 60.0) {
            std::ostringstream o; o << std::fixed << std::setprecision(0)
                                     << seconds << "s"; return o.str();
        }
        if (seconds < 3600.0) {
            int m = static_cast<int>(seconds / 60.0);
            int s = static_cast<int>(seconds - m * 60);
            std::ostringstream o; o << m << "m " << s << "s"; return o.str();
        }
        int h = static_cast<int>(seconds / 3600.0);
        int m = static_cast<int>((seconds - h * 3600.0) / 60.0);
        std::ostringstream o; o << h << "h" << m << "m"; return o.str();
    }

public:
    ProgressBar(int total, const std::string& prefix = "Progress", int barWidth = 50, bool enabled = true)
        : total(total), current(0), barWidth(barWidth), prefix(prefix), enabled(enabled),
          completed(false), last_percentage(-1), started(false) {}

    void update(int current) {
        if (!enabled) return;
        this->current = current;

        // Skip if we've already completed the progress bar
        if (completed) return;

        // Lazily start the wall-clock timer on first real update.
        if (!started) {
            start_time = std::chrono::steady_clock::now();
            started = true;
        }

        // Calculate percentage
        float percent = static_cast<float>(current) / total;
        int percentage = static_cast<int>(percent * 100);

        // Only update display if percentage has changed (prevents duplicate output)
        if (percentage == last_percentage && current < total) {
            return;
        }
        last_percentage = percentage;

        int filledWidth = static_cast<int>(barWidth * percent);

        // Build progress bar string
        std::cerr << "\r" << prefix << " [";
        for (int i = 0; i < barWidth; ++i) {
            if (i < filledWidth) std::cerr << "=";
            else if (i == filledWidth) std::cerr << ">";
            else std::cerr << " ";
        }

        // Percentage + elapsed/ETA. ETA only meaningful once we have
        // a few percent of progress; suppress it earlier to avoid
        // garbage estimates.
        std::cerr << "] " << percentage << "%";
        if (started) {
            auto now = std::chrono::steady_clock::now();
            double elapsed_s = std::chrono::duration<double>(now - start_time).count();
            std::cerr << " | elapsed " << format_duration(elapsed_s);
            if (percent > 0.01 && current < total) {
                double eta_s = elapsed_s * (1.0 - percent) / percent;
                std::cerr << ", ETA " << format_duration(eta_s);
            }
            std::cerr << "       ";  // pad for shrinking ETA strings
        }
        std::cerr.flush();

        // Mark as completed and add newline when reaching 100%
        if (current >= total) {
            std::cerr << std::endl;
            completed = true;
        }
    }

    void finish() {
        if (enabled && !completed) {
            update(total);
        }
    }
};

// Constants
const double SIGMA_SB = 5.67051196e-8; // Stefan-Boltzmann Constant
const double ABSOLUTE_ZERO_K = 2.7;      // Cosmic microwave background temperature (K) - physical minimum
const double MAX_PHYSICAL_TEMP_K = 500.0; // Maximum physically reasonable temperature for icy bodies (K)

// ---------------------------------------------------------------------------
// Solver equilibration/accuracy fix set -- ONE gate for the whole set.
//
// The three changes it enables (surface-Newton exit criterion, removal of the
// T_geo floor, and the R1 deviation form for the interior solve) are NOT
// independent and must never be enabled separately:
//
//   * the T_geo floor pins layer 0 at the analytically exact (Qb/(eps*sigma))^
//     0.25 and thereby MASKS the Newton's +9.686548e-03 K bias -- removing the
//     floor alone exposes it;
//   * the floor's Dirichlet pin on nodes 0/1 DEFEATS the deviation form
//     (measured: arrest onset j=1 with the floor on, none with it off), so
//     shipping R1 with the floor still in place buys nothing;
//   * removing the floor creates a sub-22 K population that only the Newton's
//     ulp floor can terminate.
//
// Default OFF. With the gate off every solver path is byte-identical.
// Read once; the value is process-wide and must not vary between faces.
// ---------------------------------------------------------------------------
bool solverEqFixEnabled();

// Emit a warning if the surface-Newton step limiter's growth branch or the
// iteration cap ever fired. Both are supposed to be unreachable; silence about
// them is exactly the failure mode PTM-115/116 were.
void reportSurfaceNewtonStats();

// Record nodes the removed T_geo floor would have raised (PTM_SOLVER_EQFIX).
void noteTgeoRelease(long long would_clamp, long long tested);

// Helper functions
double R350(double chi);
double skinDepth(double P, double kappa);
std::vector<double> spatialGrid(double zs, int m, int n, int b);
std::vector<double> generateStableGrid(double skin_depth, int m, int n, int b, bool verbose = false);
double albedoVar(double A0, double a, double b, double i);
double T_radeq(const Planet& planet, double lat);
double T_eq(const Planet& planet, double lat);
std::vector<double> heatCapacity(const Planet& planet, const std::vector<double>& T);
std::vector<double> thermCond(const std::vector<double>& kc, const std::vector<double>& T, double R350);
void surfTemp(Profile& p, double Qs);
void botTemp(Profile& p, double Qb);
double getTimeStep(const Profile& p, const Planet& planet, double day,
                   const Config& config, bool verbose = false);

// Configuration class
class Config {
public:
    // Physical constants
    double sigma = SIGMA_SB;  // Stefan-Boltzmann Constant
    double S0 = 1361.0;       // Solar constant at 1 AU [W.m-2]
    double chi = 2.7;         // Radiative conductivity parameter [Mitchell and de Pater, 1994]
    
    // Numerical parameters
    double F = 0.5;           // Fourier Mesh Number, must be <= 0.5 for stability
    int m = 10;               // Number of layers in upper skin depth
    int n = 15;               // Layer growth factor (only used by legacy non-adaptive grid path)
    // b = skin depths to bottom layer. With the radiative-conductivity term
    // χ(T/350)³ ~doubling K at deep regolith temperatures, the effective deep
    // skin depth is ~1.5–2× the surface lattice value. b=15 places the bottom
    // at ~9 effective e-folds of damping (~45 cm on the Moon), giving a
    // residual diurnal amplitude < 0.1 K at the bottom Neumann BC — well
    // attenuated, with ~1.7× lower thermal-kernel cost than b=20. The legacy
    // default b=5 left tens of K of residual diurnal amplitude at the bottom
    // layer, which contaminated extrapolated location-export depths.
    int b = 15;

    // Adaptive grid controls
    bool   ADAPTIVE_GRID = true;     // Use per-profile adaptive grid instead of fixed geometric
    double R_MAX         = 2.0;     // Max per-step growth factor for dz (geometric stretching cap)
    int    MIN_LAYERS_DEEP_PER_SKIN = 1;    // Minimum layers per skin depth in the deep
    double RES_DECAY_SKINS = 3.0;    // e-folding depth (in skin depths) for resolution decay
    
    // Accuracy of temperature calculations
    double DTSURF = 0.1;      // Surface temperature accuracy [K]
    int NYEARSEQ = 1;         // Equilibration time [orbits]
    int NPERDAY = 24;         // Minimum number of time steps per diurnal cycle
    // Crank-Nicolson is unconditionally stable; the Fourier-stable config dt
    // is wildly conservative for the implicit scheme. These substep targets
    // are set so the per-layer Fourier number stays below ~1 at the finest
    // depth layer, which keeps CN's low-order oscillatory error under ~1 K
    // per src/cpp/tools/test_equilibration_dt.cpp. 0 = disable override and
    // fall back to the Fourier-stable config dt.
    int EQUILIBRATION_STEPS_PER_DAY = 192;  // target steps-per-day in fastEquilibrate
    int MAIN_LOOP_STEPS_PER_DAY     = 768;  // target steps-per-day in Model::run main loop
    double DTBOT = 0.1;       // Bottom layer temperature accuracy [K]
    
    // Method to get R350
    double getR350() const { return R350(chi); }

    // Regolith conductivity law. DEFAULT MUST REMAIN Hayne2017: every result in
    // this repository was produced with it, and the byte-identity regression
    // depends on the default path being untouched.
    //
    // This lives on Config, NOT on Planet, and that is deliberate. Config is
    // built once in main and PASSED to every solver entry point, whereas every
    // solver entry point REBUILDS its own `thermal::Planet(body_name)` and so
    // discards anything the config put on the main-scope Planet. That is the
    // exact mechanism of BUGS.md PTM-53, which made `ks_override`/`kd_override`
    // silent no-ops on the Moon/Europa mesh+GPU path while still printing an
    // override banner. Putting the selector on Config sidesteps it by
    // construction; engagement is still proven by byte-difference, not assumed.
    ConductivityModel conductivity_model = ConductivityModel::Hayne2017;

    // Material property update interval
    int MATERIAL_UPDATE_INTERVAL = 1;

    // Configurable solver parameters
    double RELAXATION_FACTOR = 0.7;
    int MAX_ITERATIONS = 10;
    double TEMP_TOLERANCE = 0.1;
    double MIN_DERIVATIVE = 1e-10;

    // Solver control and adaptive stepping
    bool IMPLICIT_ALL = true;            // Force implicit (Thomas) for all layer counts
    int  THOMAS_LIMIT = 4096;            // Also use Thomas up to this size if IMPLICIT_ALL == false

    bool MULTIRATE_ADAPTIVE_STEP = true; // Enable adaptive dt using surface-controlled criterion
    double SURF_DMAX = 0.1;              // Target max |ΔT_surface| per step [K]
    double DT_GROW = 1.0;                // Factor to grow dt_scale when changes are small
    double DT_SHRINK = 0.1;              // Factor to shrink dt_scale when changes are large
    double MAX_DT_SCALE = 8.0;           // Cap for dt scaling (relative to base dt)

    // Event-aware dt control for sharp flux transitions (e.g., eclipses)
    bool  EVENT_AWARE_DT     = true;   // enable derivative-based dt refinement
    double FLUX_DERIV_THRESH = 0.0005;    // |dQ/dt| threshold [W m^-2 s^-1] to trigger refinement
    double FLUX_DQ_TARGET    = 0.1;    // target max |ΔQ| per step [W m^-2]
    double MIN_EVENT_DT      = 0.25;   // minimum dt during events [s]

    // Custom properties storage
    std::map<std::string, std::string> custom_properties;
    
    // Get a custom property with default value if not found
    std::string getProperty(const std::string& key, const std::string& default_value = "") const {
        auto it = custom_properties.find(key);
        return (it != custom_properties.end()) ? it->second : default_value;
    }
    
    // Get a custom property as integer
    int getPropertyInt(const std::string& key, int defaultValue = 0) const {
        auto it = custom_properties.find(key);
        return (it != custom_properties.end()) ? std::stoi(it->second) : defaultValue;
    }
    
    // Get a custom property as boolean
    bool getPropertyBool(const std::string& key, bool defaultValue = false) const {
        auto it = custom_properties.find(key);
        return (it != custom_properties.end()) ? (it->second == "true") : defaultValue;
    }
    
    // Get a custom property as double
    double getPropertyDouble(const std::string& key, double defaultValue = 0.0) const {
        auto it = custom_properties.find(key);
        return (it != custom_properties.end()) ? std::stod(it->second) : defaultValue;
    }
    
    // Set a custom property
    void setProperty(const std::string& key, const std::string& value) {
        custom_properties[key] = value;
    }
    
    // Set a custom property as integer
    void setProperty(const std::string& key, int value) {
        custom_properties[key] = std::to_string(value);
    }
    
    // Set a custom property as boolean
    void setProperty(const std::string& key, bool value) {
        custom_properties[key] = value ? "true" : "false";
    }
    
    // Set a custom property as double
    void setProperty(const std::string& key, double value) {
        custom_properties[key] = std::to_string(value);
    }

    // Debug output configuration
    bool THERMAL_DEBUG = false;              // Enable thermal debug output for equatorial face
    std::string DEBUG_OUTPUT_DIR = "";       // Output directory for debug file (empty = /tmp)
    double DEBUG_LAT_THRESHOLD = 10.0;       // Maximum latitude (degrees) for debug face selection
};

// Build per-profile adaptive grid using local κ(z) and diurnal frequency
// Ensures: (1) dz never violates stability proxy sqrt(kappa*dt/F), (2) dz/δ grows with depth,
// (3) total depth ≈ b·δ_surface, (4) smooth geometric stretching limited by R_MAX.
std::vector<double> generateAdaptiveGrid(const Planet& planet, const Config& config, bool verbose = false);

// Auto-tune adaptive stepping and event-aware dt using physics + flux series
// Optional configuration via custom properties:
//   "AUTO_TUNE_STEPPING" (bool, default true)
//   "SURF_STEP_FRAC"     (double, default 0.01)  // fraction of diurnal ΔT for per-step ΔT
//   "MIN_DT_FRAC_TAU"    (double, default 0.2)   // min event dt as fraction of τ_eff
void autoTuneTimeStepping(const Planet& planet,
                          double lat,
                          const Profile& profile,
                          Config& cfg,
                          const std::vector<double>& flux_times = {},
                          const std::vector<double>& flux_values = {},
                          bool verbose = false);

// Planet class (simplified from Python's planets module)
class Planet {
public:
    // Basic properties
    std::string name;
    double albedo;
    double emissivity;
    double day;       // Day length in seconds
    double year;      // Year length in seconds
    double rAU;       // Distance from Sun in AU
    double S;         // Solar constant at planet's orbit
    double Qb;        // Internal heat flux
    double ks;        // Surface thermal conductivity
    double kd;        // Deep thermal conductivity
    double rhos;      // Surface density
    double rhod;      // Deep density
    double H;         // Scale height for density/conductivity change
    double cp0;       // Reference heat capacity
    std::vector<double> cpCoeff; // Heat capacity polynomial coefficients
    std::vector<double> albedoCoef; // Albedo coefficients
    double body_radius_km = 0.0; // Mean body radius (km), 0 = use mesh as-is

    // Constructor with name or explicit properties
    Planet(const std::string& name = "Europa");

    // Constructor that takes PlanetaryBody enum
    Planet(PlanetaryBody body);

    // Constructor from ThermophysicalProperties directly
    Planet(const std::string& name, const ThermophysicalProperties& props);

    // Set custom properties
    void setProperty(const std::string& property_name, double value);
    
    // New method to set vector properties
    void setVectorProperty(const std::string& property_name, const std::vector<double>& values);
    
    // Add albedo map support
    std::shared_ptr<ptm::SpatialMap> albedo_map;
    std::shared_ptr<ptm::SpatialMap> thermal_inertia_map;
    bool using_albedo_map = false;
    bool using_thermal_inertia_map = false;

    // Directional-hemispheric (Hapke) albedo model, OFF by default. When on,
    // getAlbedoWithIncidence() returns the analytic closed form
    //   A_h(mu0) = 1 - gamma * H(mu0, w),  gamma = sqrt(1 - w),
    //   H(mu0, w) = (1 + 2*mu0) / (1 + 2*mu0*gamma),  mu0 = cos(incidence),
    // instead of the Hayne (2017) polynomial. Gated so baseline runs are
    // byte-identical when off. use_hapke_albedo defaults false; hapke_albedo_w
    // is the single-scattering albedo (set from config, or derived from A0).
    bool use_hapke_albedo = false;
    double hapke_albedo_w = -1.0;

    // Per-face H-parameter override (Diviner-derived map or user custom).
    // Empty when no map is loaded. Indexed by face index in mesh order.
    std::vector<double> face_H_overrides;
    std::vector<std::uint8_t> face_H_valid;
    inline double H_for_face(int face_idx) const {
        if (face_idx >= 0 && face_idx < static_cast<int>(face_H_overrides.size())) {
            return face_H_overrides[face_idx];
        }
        return H;
    }
    
    // Methods to load maps
    bool loadAlbedoMap(const std::string& filename, bool verbose = true);
    bool loadThermalInertiaMap(const std::string& filename, bool verbose = true);
    
    // Methods to get mapped values at a location
    double getAlbedo(double lon, double lat) const;
    double getThermalInertia(double lon, double lat) const;

    // Compute incidence-angle-dependent albedo using Hayne et al. (2017) formula:
    // A(theta) = A0 + a*(theta/(pi/4))^3 + b*(theta/(pi/2))^8
    // where theta is solar incidence angle (radians), A0 is base albedo,
    // and a, b are coefficients from albedoCoef[0], albedoCoef[1]
    // If using_albedo_map is true, returns the map value unchanged (no angle correction)
    double getAlbedoWithIncidence(double lon, double lat, double incidence_angle) const;

    // Per-face variant: A0 supplied directly (bypasses the internal spatial-map
    // lookup), same angular model. Lets the CPU mesh flux path compose the
    // incidence albedo on each face's own Bond albedo A0.
    double getAlbedoWithIncidence(double A0, double incidence_angle) const;

private:
    // Shared angular model used by both getAlbedoWithIncidence overloads.
    double albedoWithIncidenceFromA0(double A0, double incidence_angle) const;

public:
};

// Profile class
class Profile {
public:
    Planet planet;
    double emissivity;
    double chi;
    double R350_val;
    // Conductivity law in force for this profile (copied from Config at
    // construction so the hot loops need no Config lookup).
    ConductivityModel cond_model = ConductivityModel::Hayne2017;
    int nlayers;
    double day;
    const Config* config;

private:
    // Structure for aligned thermal grid data with cache-friendly layout
    struct ThermalGrid {
        aligned_vector<double> z;     // Depths
        aligned_vector<double> dz;    // Depth increments
        aligned_vector<double> d3z;   // dz[i+1]*dz[i]*(dz[i+1]+dz[i])
        aligned_vector<double> g1;    // First grid coefficient
        aligned_vector<double> g2;    // Second grid coefficient
        aligned_vector<double> kc;    // Lattice/base conductivity
        aligned_vector<double> rho;   // Density
        aligned_vector<double> T;     // Temperature
        aligned_vector<double> cp;    // Heat capacity
        aligned_vector<double> k;     // Thermal conductivity
        aligned_vector<double> kappa; // Thermal diffusivity

        void resize(size_t size) {
            z.resize(size);
            dz.resize(size > 0 ? size - 1 : 0);
            d3z.resize(size > 0 ? size - 2 : 0);
            g1.resize(size > 0 ? size - 2 : 0);
            g2.resize(size > 0 ? size - 2 : 0);
            kc.resize(size);
            rho.resize(size);
            T.resize(size);
            cp.resize(size);
            k.resize(size);
            kappa.resize(size);
        }

        void reserve(size_t capacity) {
            z.reserve(capacity);
            dz.reserve(capacity > 0 ? capacity - 1 : 0);
            d3z.reserve(capacity > 0 ? capacity - 2 : 0);
            g1.reserve(capacity > 0 ? capacity - 2 : 0);
            g2.reserve(capacity > 0 ? capacity - 2 : 0);
            kc.reserve(capacity);
            rho.reserve(capacity);
            T.reserve(capacity);
            cp.reserve(capacity);
            k.reserve(capacity);
            kappa.reserve(capacity);
        }
    };

    // Memory pool for temporary arrays used in calculations
    struct MemoryPool {
        aligned_vector<double> a_buffer;
        aligned_vector<double> b_buffer;
        aligned_vector<double> c_buffer;
        aligned_vector<double> d_buffer;

        void ensure_capacity(size_t n) {
            if (a_buffer.size() < n) {
                a_buffer.resize(n);
                b_buffer.resize(n);
                c_buffer.resize(n);
                d_buffer.resize(n);
            }
        }

        double* a() { return a_buffer.data(); }
        double* b() { return b_buffer.data(); }
        double* c() { return c_buffer.data(); }
        double* d() { return d_buffer.data(); }
    };

    // Material counter for updating material thermophysical properties
    size_t material_counter = 0;

    ThermalGrid grid;
    MemoryPool memory_pool;

public:
    ThermalGrid& getGrid() { return grid; }
    const ThermalGrid& getGrid() const { return grid; }

    // Accessors to maintain API compatibility (aligned_vector)
    aligned_vector<double>& z()     { return grid.z; }
    aligned_vector<double>& dz()    { return grid.dz; }
    aligned_vector<double>& d3z()   { return grid.d3z; }
    aligned_vector<double>& g1()    { return grid.g1; }
    aligned_vector<double>& g2()    { return grid.g2; }
    aligned_vector<double>& kc()    { return grid.kc; }
    aligned_vector<double>& rho()   { return grid.rho; }
    aligned_vector<double>& T()     { return grid.T; }
    aligned_vector<double>& cp()    { return grid.cp; }
    aligned_vector<double>& k()     { return grid.k; }
    aligned_vector<double>& kappa() { return grid.kappa; }

    // Const accessors
    const aligned_vector<double>& z() const     { return grid.z; }
    const aligned_vector<double>& dz() const    { return grid.dz; }
    const aligned_vector<double>& d3z() const   { return grid.d3z; }
    const aligned_vector<double>& g1() const    { return grid.g1; }
    const aligned_vector<double>& g2() const    { return grid.g2; }
    const aligned_vector<double>& kc() const    { return grid.kc; }
    const aligned_vector<double>& rho() const   { return grid.rho; }
    const aligned_vector<double>& T() const     { return grid.T; }
    const aligned_vector<double>& cp() const    { return grid.cp; }
    const aligned_vector<double>& k() const     { return grid.k; }
    const aligned_vector<double>& kappa() const { return grid.kappa; }

    // Constructor.
    // H_override < 0 means "use planet.H"; otherwise it overrides the
    // body-default H-parameter for both the depth-grid skin-depth estimate
    // and the rho(z)/k(z) exp(-z/H) initialization. Per-face H maps go through
    // this path (driver looks up planet.H_for_face(face_idx)).
    Profile(const Planet& planet = Planet(), double lat = 0.0,
            const Config& config = Config(), bool verbose = false,
            double H_override = -1.0);

    // Methods
    void init_T(const Planet& planet, double lat, bool verbose = false);
    void update_cp();
    void update_k();
    void update_T(double dt, double Qs = 0, double Qb = 0);

    // Utility updaters
    void recomputeKappa() {
        const size_t n = k().size();
        auto& kk = kappa();
        const auto& kr = k();
        const auto& rr = rho();
        const auto& cc = cp();
        if (kk.size() != n) kk.resize(n);
        #pragma omp parallel for schedule(static) if(n > 1024)
        for (size_t i = 0; i < n; ++i) {
            kk[i] = kr[i] / (rr[i] * cc[i]);
        }
    }

    void updateThermalProperties(bool update_k_first = true) {
        if (update_k_first) {
            update_k();
            update_cp();
        } else {
            update_cp();
            update_k();
        }
        recomputeKappa();
    }
};

// Model class
class Model {
public:
    Planet planet;
    double lat;
    double Sabs;
    double r;
    double nu;
    double nudot;
    double dec;
    double Qs;
    
    // Time tracking
    double t;
    double dt;
    double dtout;
    double equiltime;
    double endtime;
    
    // Arrays for flux data
    Eigen::MatrixXd flux_arr;
    
    // Output arrays
    Eigen::MatrixXd T;
    Eigen::VectorXd lt;
    
    // Number of steps and layers
    int N_steps;
    int N_z;
    
    // Profile object
    Profile profile;
    
    // Tuned configuration owned by the model (mutable)
    Config tuned_config;
    // Config pointer used throughout (points to tuned_config)
    const Config* config;
    
    // Store explicit output time points
    std::vector<double> output_time_points;
    
    // Equilibration control
    bool use_full_time_series;
    int equilibration_steps;
    int main_simulation_steps;
    
    // Constructor
    Model(const Planet& planet = Planet(), double lat = 0.0, 
          const std::vector<double>& output_times = std::vector<double>(),
          const Eigen::MatrixXd& flux_arr = Eigen::MatrixXd(), 
          const Config& config = Config(),
          bool show_progress = true);
    
    // Methods
    void run();
    void advance();
    void updateOrbit();
    void surfFlux();
    void fastEquilibrate(int days, double target_time);
    
    // Add this new method
    void setEquilibrationSteps(int steps) {
        equilibration_steps = steps;
        if (equilibration_steps > 0) {
            use_full_time_series = true;
        }
    }
    
    // Add this new method
    void setStartTime(double startTime) {
        t = startTime;  // Explicitly set the starting time
    }
    
    // Progress bar flag
    bool show_progress;

protected:
    // Method to interpolate flux from arrays
    double interpolateFlux(double time);
    std::vector<double> flux_times;
    std::vector<double> flux_values;
    size_t flux_idx = 1;  // rolling index for monotonic time

    // Store original flux epoch for diagnostics (after we shift times to start at 0)
    double flux_time_offset = 0.0;

    // Adaptive time step control
    double dt_scale = 1.0;                       // multiplicative dt scaling
    double last_surface_T = std::numeric_limits<double>::quiet_NaN();
    double chooseAdaptiveDt(double target_time); // compute next dt considering stability/accuracy
    double estimateFluxDerivative(double time);          // finite-difference dQ/dt from series
    double eventLimitedDt(double base_dt, double target_time); // clamp dt when |dQ/dt| is large
};

// Function to automatically optimize grid parameters based on physical properties
void calculateOptimalGridParameters(const Planet& planet, Config& config);

// =============================================================================
// SIMD Batch Thermal Solver (SIMD-over-Pixels)
// =============================================================================

// SIMD width constants - always use 8 for array sizing, runtime check for actual width
constexpr int THERMAL_BATCH_SIZE = 8;     // 8 doubles = 512 bits (AVX-512)
constexpr int MAX_THERMAL_LAYERS = 128;   // Maximum supported depth layers

/**
 * Structure of Arrays (SoA) for batch thermal solving.
 *
 * Memory layout: data[layer * THERMAL_BATCH_SIZE + face_in_batch]
 *
 * This enables SIMD operations across 8 faces at each depth layer:
 *   Layer 0: [face0, face1, face2, face3, face4, face5, face6, face7]  <- One AVX-512 load
 *   Layer 1: [face0, face1, face2, face3, face4, face5, face6, face7]  <- One AVX-512 load
 *   ...
 *
 * The Thomas algorithm is sequential in depth (layer i depends on layer i-1),
 * but 8 independent faces can be solved in parallel at each depth.
 */
struct ThermalBatchSoA {
    int n_layers;            // Number of depth layers (same for all faces in batch)
    int n_faces;             // Actual faces in batch (may be < THERMAL_BATCH_SIZE)

    // Temperature profile [n_layers * THERMAL_BATCH_SIZE]
    // Layout: T[layer * 8 + face_in_batch]
    alignas(64) double T[MAX_THERMAL_LAYERS * THERMAL_BATCH_SIZE];

    // Material properties [n_layers * THERMAL_BATCH_SIZE]
    alignas(64) double cp[MAX_THERMAL_LAYERS * THERMAL_BATCH_SIZE];
    alignas(64) double k[MAX_THERMAL_LAYERS * THERMAL_BATCH_SIZE];
    alignas(64) double kappa[MAX_THERMAL_LAYERS * THERMAL_BATCH_SIZE];
    alignas(64) double rho[MAX_THERMAL_LAYERS * THERMAL_BATCH_SIZE];
    alignas(64) double kc[MAX_THERMAL_LAYERS * THERMAL_BATCH_SIZE];   // Base conductivity

    // Thomas algorithm buffers [n_layers * THERMAL_BATCH_SIZE]
    alignas(64) double a_coef[MAX_THERMAL_LAYERS * THERMAL_BATCH_SIZE];   // Sub-diagonal
    alignas(64) double b_coef[MAX_THERMAL_LAYERS * THERMAL_BATCH_SIZE];   // Main diagonal
    alignas(64) double c_coef[MAX_THERMAL_LAYERS * THERMAL_BATCH_SIZE];   // Super-diagonal
    alignas(64) double d_coef[MAX_THERMAL_LAYERS * THERMAL_BATCH_SIZE];   // RHS / solution

    // Grid coefficients [n_layers] - SHARED across all faces in batch
    alignas(64) double z[MAX_THERMAL_LAYERS];       // Depth positions
    alignas(64) double dz[MAX_THERMAL_LAYERS];      // Layer spacing (n_layers-1 used)
    alignas(64) double g1[MAX_THERMAL_LAYERS];      // Grid coefficient 1
    alignas(64) double g2[MAX_THERMAL_LAYERS];      // Grid coefficient 2

    // Surface properties [THERMAL_BATCH_SIZE]
    alignas(64) double emissivity[THERMAL_BATCH_SIZE];
    alignas(64) double flux[THERMAL_BATCH_SIZE];    // Surface absorbed flux
    alignas(64) double Qb[THERMAL_BATCH_SIZE];      // Bottom heat flux

    // Initial temperature for equilibrium [THERMAL_BATCH_SIZE]
    alignas(64) double T_eq_init[THERMAL_BATCH_SIZE];

    // Material parameters (shared across all faces in batch)
    double cp0;              // Base heat capacity
    double R350_val;         // Radiative conductivity coefficient (Hayne 2017)
    // Conductivity law for this batch. Default Hayne2017 so a batch that is
    // zero-initialized or built by older code keeps the historical behavior.
    ConductivityModel cond_model = ConductivityModel::Hayne2017;
    double rho_s;            // Surface density
    double rho_d;            // Deep density
    double k_s;              // Surface conductivity
    double k_d;              // Deep conductivity
    double H_val;            // e-folding depth for density/conductivity
    int n_cp_coeffs;         // Number of cp polynomial coefficients
    alignas(64) double cp_coeffs[8];  // cp(T) polynomial coefficients

    // Physical constants
    double sigma_sb;         // Stefan-Boltzmann constant

    // Accessors for layer-indexed SIMD loads
    inline double* layer_ptr(double* arr, int layer_idx) {
        return &arr[layer_idx * THERMAL_BATCH_SIZE];
    }
    inline const double* layer_ptr(const double* arr, int layer_idx) const {
        return &arr[layer_idx * THERMAL_BATCH_SIZE];
    }

    // Initialize batch with grid and material parameters
    void initialize(int layers, int faces, const Planet& planet, const Config& config);

    // Copy grid from existing Profile.
    // If H_per_face is non-null, it must point to `n_faces` doubles; rho/kc
    // are then initialized per-face with each face's own H. Otherwise the
    // batch's shared H_val is used (legacy behavior).
    void copyGridFromProfile(const Profile& ref_profile,
                             const double* H_per_face = nullptr);

    // Set surface flux values for all faces in batch (scattered input)
    void setFlux(const double* flux_values, int num_faces);

    // Set surface flux values from contiguous memory (SIMD optimized)
    // Use this when flux_values points to 8 contiguous doubles (column-major layout)
    void setFluxContiguous(const double* flux_values, int num_faces);

    // Advance one timestep for all faces in batch
    void advanceTimestep(double dt);

    // Get surface temperature for a specific face
    double getSurfaceTemp(int face_idx) const { return T[face_idx]; }

    // Get temperature at specific layer and face
    double getTemp(int layer, int face_idx) const {
        return T[layer * THERMAL_BATCH_SIZE + face_idx];
    }
};

// Batch solver helper functions
void initializeBatchGrid(ThermalBatchSoA& batch, const std::vector<double>& z_grid,
                         const Planet& planet);
void updateMaterialPropertiesBatchSIMD(ThermalBatchSoA& batch);
void buildTridiagonalSystemBatch(ThermalBatchSoA& batch, double dt);
void solveThomasBatchSIMD(ThermalBatchSoA& batch);
void applySurfaceBCBatch(ThermalBatchSoA& batch, double dt);
void applyBottomBCBatch(ThermalBatchSoA& batch);

} // namespace thermal

#endif // THERMAL_MODEL_H