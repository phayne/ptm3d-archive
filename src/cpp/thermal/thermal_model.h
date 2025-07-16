#ifndef THERMAL_MODEL_H
#define THERMAL_MODEL_H

#include <vector>
#include <cmath>
#include <string>
#include <memory>
#include <iostream>  // Add this for std::cout
#include <Eigen/Dense>
#include <map>       // Add this for std::map
#include "planetary_properties.h"  // Add this include
#include "../maps/spatial_map.h"  // Add this include for SpatialMap class

namespace thermal {

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
    bool completed; // Add this flag to track completion
    
public:
    ProgressBar(int total, const std::string& prefix = "Progress", int barWidth = 50, bool enabled = true) 
        : total(total), current(0), barWidth(barWidth), prefix(prefix), enabled(enabled), completed(false) {}
    
    void update(int current) {
        if (!enabled) return;
        this->current = current;
        
        // Skip if we've already completed the progress bar
        if (completed) return;
        
        // Calculate percentage
        float percent = static_cast<float>(current) / total;
        int filledWidth = static_cast<int>(barWidth * percent);
        
        // Build progress bar string
        std::cout << "\r" << prefix << " [";
        for (int i = 0; i < barWidth; ++i) {
            if (i < filledWidth) std::cout << "=";
            else if (i == filledWidth) std::cout << ">";
            else std::cout << " ";
        }
        
        // Add percentage at the end
        std::cout << "] " << static_cast<int>(percent * 100) << "%";
        std::cout.flush();
        
        // Mark as completed and add newline when reaching 100%
        if (current >= total) {
            std::cout << std::endl;
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

// Helper functions
double R350(double chi);
double skinDepth(double P, double kappa);
std::vector<double> spatialGrid(double zs, int m, int n, int b);
double albedoVar(double A0, double a, double b, double i);
double T_radeq(const Planet& planet, double lat);
double T_eq(const Planet& planet, double lat);
std::vector<double> heatCapacity(const Planet& planet, const std::vector<double>& T);
std::vector<double> thermCond(const std::vector<double>& kc, const std::vector<double>& T, double R350);
void surfTemp(Profile& p, double Qs);
void botTemp(Profile& p, double Qb);
double getTimeStep(const Profile& p, const Planet& planet, double day, const Config& config, bool verbose = false);

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
    int n = 5;                // Layer increase with depth: dz[i] = dz[i-1]*(1+1/n)
    int b = 20;               // Number of skin depths to bottom layer
    
    // Accuracy of temperature calculations
    double DTSURF = 0.1;      // Surface temperature accuracy [K]
    int NYEARSEQ = 1;         // Equilibration time [orbits]
    int NPERDAY = 24;         // Minimum number of time steps per diurnal cycle
    double DTBOT = 0.1;       // Bottom layer temperature accuracy [K]
    
    // Method to get R350
    double getR350() const { return R350(chi); }

    // New configurable solver parameters
    double RELAXATION_FACTOR = 0.7;      // Relaxation factor for Newton-Raphson (0-1)
    int MAX_ITERATIONS = 30;             // Maximum iterations for surface temperature solver
    double TEMP_TOLERANCE = 0.01;        // Temperature convergence tolerance (K)
    double MIN_DERIVATIVE = 1e-10;       // Minimum allowed derivative to prevent division by near-zero

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
};

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
    
    // Constructor with name or explicit properties
    Planet(const std::string& name = "Europa");
    
    // New constructor that takes PlanetaryBody enum
    Planet(PlanetaryBody body);
    
    // New method to set custom properties
    void setProperty(const std::string& property_name, double value);
    
    // New method to set vector properties
    void setVectorProperty(const std::string& property_name, const std::vector<double>& values);
    
    // Add albedo map support
    std::shared_ptr<ptm::SpatialMap> albedo_map;
    std::shared_ptr<ptm::SpatialMap> thermal_inertia_map;
    bool using_albedo_map = false;
    bool using_thermal_inertia_map = false;
    
    // Methods to load maps
    bool loadAlbedoMap(const std::string& filename, bool verbose = true);
    bool loadThermalInertiaMap(const std::string& filename, bool verbose = true);
    
    // Methods to get mapped values at a location
    double getAlbedo(double lon, double lat) const;
    double getThermalInertia(double lon, double lat) const;
};

// Profile class
class Profile {
public:
    Planet planet;
    double emissivity;
    double chi;
    double R350_val;
    
    // Vectors for spatial grid and thermal properties
    std::vector<double> z;
    std::vector<double> dz;
    std::vector<double> d3z;
    std::vector<double> g1;
    std::vector<double> g2;
    std::vector<double> kc;
    std::vector<double> rho;
    std::vector<double> T;
    std::vector<double> cp;
    std::vector<double> k;
    
    int nlayers;
    double day;  // Add day length for time step calculation
    
    // Constructor
    Profile(const Planet& planet = Planet(), double lat = 0.0, 
            const Config& config = Config(), bool verbose = false);
    
    // Methods
    void init_T(const Planet& planet, double lat, bool verbose = false);
    void update_cp();
    void update_k();
    void update_T(double dt, double Qs = 0, double Qb = 0);
    
    // Reference to config
    const Config* config;
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
    
    // Config reference
    const Config* config;
    
    // Add this member to store explicit output time points
    std::vector<double> output_time_points;
    
    // Add these new members
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

private:
    // Method to interpolate flux from arrays
    double interpolateFlux(double time);
    
    // Store the flux array as separate time and value vectors
    std::vector<double> flux_times;
    std::vector<double> flux_values;
};

// Function to automatically optimize grid parameters based on physical properties
void calculateOptimalGridParameters(const Planet& planet, Config& config);

} // namespace thermal

#endif // THERMAL_MODEL_H

