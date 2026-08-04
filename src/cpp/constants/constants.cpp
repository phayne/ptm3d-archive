#include "constants.h"
#include <cmath>
#include <iostream>
#include <string>

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
// PTM-43: each primary body's (TEMPERATURE, IR_FLUX) pair must satisfy
// IR_FLUX = sigma * TEMPERATURE^4, because the flux is consumed as an
// EMITTER-SURFACE flux (see PTM-40 and calculatePrimaryHeating). The pair is
// the unit of correctness -- you cannot tell which half is wrong by looking at
// either half alone -- so all four pairs were corrected together and are now
// checked at load time by checkPrimaryBodyConstants() at the bottom of this
// file.
const double JUPITER_RADIUS = 69911.0;            // km
const double JUPITER_MASS = 1.8982e27;            // kg
// PTM-43: was 165.0 K, which is Jupiter's ~1-BAR temperature mislabeled
// "effective temperature". Jupiter's T_eff is 124.4 K (Hanel et al. 1981), and
// 124.4 is what the IR flux below has always implied. The flux was right and
// the temperature was wrong. No numeric effect: this constant is read only in
// a branch of calculatePrimaryHeating that JUPITER can never reach.
const double JUPITER_TEMPERATURE = 124.4;         // K (effective temperature)
const double JUPITER_BOND_ALBEDO = 0.343;         // Bond albedo
// EMITTER-SURFACE flux: sigma*T_eff^4 for T_eff = 124.4 K. NOT the irradiance
// at Europa -- that is this value times the dA->sphere view factor sin^2(alpha)
// ~= 0.0109, i.e. ~0.15 W/m^2 at the sub-Jovian point. The old "13.8 W/m^2 at
// Europa" comment invited exactly the confusion that hid PTM-40.
const double JUPITER_IR_FLUX = 13.8;              // W/m^2 leaving Jupiter's surface

// ---- Earth Constants ----
const double EARTH_RADIUS = 6371.0;               // km
const double EARTH_MASS = 5.972e24;               // kg
const double EARTH_TEMPERATURE = 255.0;           // K (effective temperature)
const double EARTH_BOND_ALBEDO = 0.306;           // Bond albedo
// EMITTER-SURFACE flux: sigma*T_eff^4 for T_eff = 255 K (Earth's mean OLR).
// The old "at Earth distance from Sun" comment was wrong twice over -- the solar
// constant there is 1361, not 240 -- but the VALUE is correct for its use.
// PTM-43: the only pair that was already self-consistent (240 vs
// sigma*(255 K)^4 = 239.76, ratio 1.001). Unchanged.
const double EARTH_IR_FLUX = 240.0;               // W/m^2 leaving Earth's surface

// ---- Mars Constants ----
const double MARS_RADIUS = 3389.5;                // km
const double MARS_MASS = 6.4171e23;            // kg
const double MARS_TEMPERATURE = 210.0;            // K (effective temperature)
const double MARS_BOND_ALBEDO = 0.25;            // Bond albedo
// PTM-43: was 70.0, which is 37 % below sigma*(210 K)^4 = 110.3 and matches no
// stated temperature. Here the TEMPERATURE was defensible and the flux was not.
// EMITTER-SURFACE flux, like the others.
const double MARS_IR_FLUX = 110.3;               // W/m^2 leaving Mars's surface

// ---- Saturn Constants ----
const double SATURN_RADIUS = 58232.0;             // km
const double SATURN_MASS = 5.6834e26;             // kg
// PTM-43: was 134.0 K. Saturn's effective temperature is 95 K (Hanel et al.
// 1983); 134 K is a 1-bar-ish value, the same mislabeling as Jupiter's 165 K.
const double SATURN_TEMPERATURE = 95.0;           // K (effective temperature)
const double SATURN_BOND_ALBEDO = 0.342;          // Bond albedo
// PTM-43: was 15.0, which is 3.25x too high and is almost exactly the SOLAR
// constant at Saturn (1361/9.58^2 = 14.83) -- a solar-constant-for-thermal-
// emission mis-entry. EMITTER-SURFACE flux = sigma*(95 K)^4.
const double SATURN_IR_FLUX = 4.62;               // W/m^2 leaving Saturn's surface

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

// ============================================================================
// PTM-43: load-time consistency check on the primary-body (T, IR_FLUX) pairs.
//
// The IR flux constants above are consumed as EMITTER-SURFACE fluxes (PTM-40's
// Omega/pi view factor requires it), so each must equal sigma * T^4 for its
// own stated temperature. Three of the four pairs did not, for years, and the
// disagreement was only ever found by someone dividing by hand:
//     JUPITER  165.0 K -> 42.03  vs flux 13.8    (67 % apart)
//     SATURN   134.0 K -> 18.28  vs flux 15.0    (18 % apart)
//     MARS     210.0 K -> 110.28 vs flux 70.0    (37 % apart)
//     EARTH    255.0 K -> 239.76 vs flux 240.0   (0.1 % -- the only good pair)
// This runs at library load, so the next mismatched pair announces itself
// instead of waiting to be noticed. It WARNS rather than aborting: a wrong
// constant should not make an unrelated binary refuse to start.
// ============================================================================
bool checkPrimaryBodyConstants(double rel_tol) {
    struct Pair { const char* name; double T; double flux; };
    const Pair pairs[] = {
        {"JUPITER", JUPITER_TEMPERATURE, JUPITER_IR_FLUX},
        {"EARTH",   EARTH_TEMPERATURE,   EARTH_IR_FLUX},
        {"SATURN",  SATURN_TEMPERATURE,  SATURN_IR_FLUX},
        {"MARS",    MARS_TEMPERATURE,    MARS_IR_FLUX},
    };
    bool all_ok = true;
    for (const auto& p : pairs) {
        if (!(p.T > 0.0)) continue;
        const double sb = STEFAN_BOLTZMANN * p.T * p.T * p.T * p.T;
        if (sb <= 0.0) continue;
        const double rel = std::fabs(p.flux - sb) / sb;
        if (rel > rel_tol) {
            all_ok = false;
            std::cerr << "[constants] WARNING: " << p.name
                      << " (temperature, IR flux) pair is inconsistent: "
                      << p.name << "_IR_FLUX = " << p.flux
                      << " W/m^2 but sigma*(" << p.T << " K)^4 = " << sb
                      << " W/m^2 (" << (rel * 100.0) << " % apart, tolerance "
                      << (rel_tol * 100.0) << " %).\n"
                      << "  These constants are EMITTER-SURFACE quantities and must "
                         "agree. One of the two is wrong; see BUGS.md PTM-43."
                      << std::endl;
        }
    }
    return all_ok;
}

namespace {
// Runs when libconstants loads, i.e. before main() in every binary that links
// it -- there is no way to forget to call it.
const bool g_primary_body_constants_ok = checkPrimaryBodyConstants(0.05);
}  // namespace

} // namespace constants
} // namespace europa_thermal