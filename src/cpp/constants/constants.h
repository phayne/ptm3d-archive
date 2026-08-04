#ifndef PTM_THERMAL_CONSTANTS_H
#define PTM_THERMAL_CONSTANTS_H

#include <string>

namespace ptm_thermal {
namespace constants {

// ---- Physical Constants ----
extern const double SPEED_OF_LIGHT;       // Speed of light in vacuum (m/s)
extern const double STEFAN_BOLTZMANN;     // Stefan-Boltzmann constant (W/m^2/K^4)
extern const double BOLTZMANN_CONSTANT;   // J/K
extern const double PLANCK_CONSTANT;      // J·s

// ---- Mathematical Constants ----
extern const double PI;                  // π
extern const double DEG_TO_RAD;          // Convert degrees to radians
extern const double RAD_TO_DEG;          // Convert radians to degrees
extern const double SECONDS_PER_DAY;     // Seconds in a day

// ---- Astronomical Constants ----
extern const double ASTRONOMICAL_UNIT;    // 1 AU in meters
extern const double AU_TO_KM;            // 1 AU in kilometers
extern const double KM_PER_AU;           // Kilometers per astronomical unit
extern const double SUN_RADIUS;          // Solar radius (km)
extern const double SOLAR_LUMINOSITY;     // Solar luminosity (W)
extern const double SOLAR_CONSTANT;       // Solar constant at 1 AU (W/m^2)
extern const double SOLAR_CONSTANT_1AU;   // Same as above, explicitly named

// ---- Jupiter Constants ----
extern const double JUPITER_RADIUS;       // Equatorial radius of Jupiter (km)
extern const double JUPITER_MASS;         // Mass of Jupiter (kg)
extern const double JUPITER_TEMPERATURE;  // Effective temperature (K)
extern const double JUPITER_BOND_ALBEDO;  // Bond albedo
extern const double JUPITER_IR_FLUX;      // IR flux from Jupiter

// ---- Earth Constants ----
extern const double EARTH_RADIUS;               // km
extern const double EARTH_MASS;                 // kg
extern const double EARTH_TEMPERATURE;          // K (effective temperature)
extern const double EARTH_BOND_ALBEDO;          // Bond albedo
extern const double EARTH_IR_FLUX;              // W/m^2 at Earth distance from Sun

// ---- Mars Constants ----
extern const double MARS_RADIUS;                // km
extern const double MARS_MASS;                  // kg
extern const double MARS_TEMPERATURE;           // K (effective temperature)
extern const double MARS_BOND_ALBEDO;           // Bond albedo
extern const double MARS_IR_FLUX;               // W/m^2 at Mars distance from Sun

// ---- Saturn Constants ----
extern const double SATURN_RADIUS;             // km
extern const double SATURN_MASS;               // kg
extern const double SATURN_TEMPERATURE;        // K (effective temperature)
extern const double SATURN_BOND_ALBEDO;        // Bond albedo
extern const double SATURN_IR_FLUX;            // W/m^2 at Saturn distance from Sun


// ---- Europa Constants ----
extern const double EUROPA_RADIUS;        // Mean radius of Europa (km)
extern const double EUROPA_MASS;          // Mass of Europa (kg)
extern const double EUROPA_ORBIT_RADIUS;  // Semi-major axis of Europa's orbit around Jupiter (km)
extern const double EUROPA_SEMI_MAJOR_AXIS; // Same as above
extern const double EUROPA_ALBEDO;        // Bond albedo of Europa
extern const double EUROPA_EMISSIVITY;    // Thermal emissivity of Europa's surface
extern const double EUROPA_SOLAR_CONSTANT;  // Solar constant at Europa's orbit (W/m²)
extern const double EUROPA_ORBITAL_PERIOD;  // Days

// ---- Generic Body Constants ----
extern const double PRIMARY_RADIUS;       // Primary radius (Jupiter)
extern const double PRIMARY_MASS;         // Primary mass (Jupiter)
extern const double PRIMARY_TEMPERATURE;  // Primary effective temperature
extern const double PRIMARY_BOND_ALBEDO;  // Primary Bond albedo
extern const double PRIMARY_IR_FLUX;      // IR flux from Primary
extern const double PRIMARY_DISTANCE;     // Distance from Europa
extern const double BODY_RADIUS;          // Radius of the body (Europa)
extern const double BODY_MASS;          // Mass of the body (Europa)
extern const double BODY_ALBEDO;         // Bond albedo of the body
extern const double BODY_EMISSIVITY;     // Emissivity of the body's surface
extern const double BODY_SOLAR_CONSTANT; // Solar constant at the body's orbit
extern const double BODY_ORBITAL_PERIOD; // Orbital period of the body

// ---- Thermal Model Parameters ----
extern const double DEFAULT_THERMAL_CONDUCTIVITY;  // W/(m·K)
extern const double DEFAULT_HEAT_CAPACITY;         // J/(kg·K)
extern const double DEFAULT_DENSITY;               // kg/m^3

// ---- Simulation Parameters ----
extern const double DEFAULT_TIMESTEP;          // Default simulation timestep (s)
extern const double DEFAULT_GRID_RESOLUTION;   // Default spatial grid resolution (m)
extern const double CONVERGENCE_TOLERANCE;     // Temperature convergence tolerance (K)
extern const double N_SOLAR_SAMPLES;          // Number of solar disk samples for illumination

// ---- SPICE Parameters ----
extern const std::string EUROPA_SPICE_ID;      // SPICE ID for Europa
extern const std::string JUPITER_SPICE_ID;     // SPICE ID for Jupiter
extern const std::string SUN_SPICE_ID;         // SPICE ID for Sun

// ---- Other constants needed for mesh_utils.cpp ----
extern const double EPSILON;              // Small value for floating-point comparisons

/**
 * @brief PTM-43: verify every primary-body (TEMPERATURE, IR_FLUX) pair against
 *        IR_FLUX == sigma * TEMPERATURE^4.
 *
 * The IR flux constants are consumed as EMITTER-SURFACE fluxes (PTM-40), so a
 * pair that disagrees means one of its two halves is wrong -- and you cannot
 * tell which by looking at either alone. Runs automatically at library load
 * and WARNS to stderr on any mismatch; exposed here so a test can call it.
 *
 * @param rel_tol fractional tolerance (0.05 = 5 %)
 * @return true if every pair agrees within rel_tol
 */
bool checkPrimaryBodyConstants(double rel_tol = 0.05);

} // namespace constants
} // namespace ptm_thermal

#endif // PTM_THERMAL_CONSTANTS_H