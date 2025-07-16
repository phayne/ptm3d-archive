#include "constants.h"
#include <cmath>

namespace ptm_thermal {
namespace constants {

// ---- Physical Constants ----
const double SPEED_OF_LIGHT = 299792458.0;        // m/s
const double STEFAN_BOLTZMANN = 5.670374419e-8;   // W/m^2/K^4
const double BOLTZMANN_CONSTANT = 1.380649e-23;   // J/K
const double PLANCK_CONSTANT = 6.62607015e-34;    // J·s

// ---- Mathematical Constants ----
const double PI = 3.14159265358979323846;         // π
const double DEG_TO_RAD = PI / 180.0;             // Convert degrees to radians
const double RAD_TO_DEG = 180.0 / PI;             // Convert radians to degrees
const double SECONDS_PER_DAY = 86400.0;           // 24*60*60

// ---- Astronomical Constants ----
const double ASTRONOMICAL_UNIT = 149597870700.0;  // m
const double AU_TO_KM = 149597870.7;              // km
constexpr double SUN_RADIUS = 695700.0;           // Sun radius in km
const double SOLAR_LUMINOSITY = 3.828e26;         // W
const double SOLAR_CONSTANT = 1361.0;             // W/m^2
constexpr double SOLAR_CONSTANT_1AU = 1361.0;     // Solar constant at 1 AU in W/m^2
constexpr double KM_PER_AU = 149597870.7;         // Kilometers per astronomical unit

// ---- Jupiter Constants ----
const double JUPITER_RADIUS = 69911.0;            // km
const double JUPITER_MASS = 1.8982e27;            // kg
const double JUPITER_TEMPERATURE = 165.0;         // K (effective temperature)
const double JUPITER_BOND_ALBEDO = 0.343;         // Bond albedo
const double JUPITER_IR_FLUX = 13.8;              // W/m^2 at Europa

// ---- Europa Constants ----
const double EUROPA_RADIUS = 1560.8;              // km
const double EUROPA_MASS = 4.8e22;                // kg
const double EUROPA_ORBIT_RADIUS = 671100.0;      // km
const double EUROPA_SEMI_MAJOR_AXIS = 671100.0;   // km (same as EUROPA_ORBIT_RADIUS)
const double EUROPA_ALBEDO = 0.67;                // Bond albedo
const double EUROPA_EMISSIVITY = 0.96;            // Typical ice emissivity
const double EUROPA_SOLAR_CONSTANT = 50.0;        // W/m² (approximate at Europa distance)
const double EUROPA_ORBITAL_PERIOD = 3.55118;     // days

// ---- Generic Body Constants ----
const double PRIMARY_RADIUS = JUPITER_RADIUS; // Primary radius (Jupiter)
const double PRIMARY_MASS = JUPITER_MASS;   // Primary mass (Jupiter)
const double PRIMARY_TEMPERATURE = JUPITER_TEMPERATURE; // Primary effective temperature
const double PRIMARY_BOND_ALBEDO = JUPITER_BOND_ALBEDO; // Primary Bond albedo
const double PRIMARY_IR_FLUX = JUPITER_IR_FLUX; // IR flux from Primary
const double PRIMARY_DISTANCE = EUROPA_ORBIT_RADIUS; // Distance from Europa
const double BODY_RADIUS = EUROPA_RADIUS; // Radius of the body (Europa)
const double BODY_MASS = EUROPA_MASS; // Mass of the body (Europa)
const double BODY_ALBEDO = EUROPA_ALBEDO; // Bond albedo of the body
const double BODY_EMISSIVITY = EUROPA_EMISSIVITY; // Emissivity of the body's surface
const double BODY_SOLAR_CONSTANT = EUROPA_SOLAR_CONSTANT; // Solar constant at the body's orbit
const double BODY_ORBITAL_PERIOD = EUROPA_ORBITAL_PERIOD; // Orbital period of the body

// ---- Thermal Model Parameters ----
const double DEFAULT_THERMAL_CONDUCTIVITY = 2.3;  // W/(m·K) - typical for ice
const double DEFAULT_HEAT_CAPACITY = 2000.0;      // J/(kg·K) - typical for ice
const double DEFAULT_DENSITY = 917.0;             // kg/m^3 - typical for ice

// ---- Simulation Parameters ----
const double DEFAULT_TIMESTEP = 3600.0;           // 1 hour in seconds
const double DEFAULT_GRID_RESOLUTION = 1000.0;    // 1 km in meters
const double CONVERGENCE_TOLERANCE = 0.01;        // 0.01 K
const double N_SOLAR_SAMPLES = 9;                 // Number of solar disk samples for illumination

// ---- SPICE Parameters ----
const std::string EUROPA_SPICE_ID = "502";        // SPICE ID for Europa
const std::string JUPITER_SPICE_ID = "599";       // SPICE ID for Jupiter
const std::string SUN_SPICE_ID = "10";            // SPICE ID for Sun

// ---- Other constants needed for calculations ----
const double EPSILON = 1.0e-10;                   // Small value for comparisons

} // namespace constants
} // namespace europa_thermal