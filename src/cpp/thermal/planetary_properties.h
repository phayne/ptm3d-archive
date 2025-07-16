#ifndef PLANETARY_PROPERTIES_H
#define PLANETARY_PROPERTIES_H

#include <string>
#include <vector>
#include <unordered_map>
#include <cmath>

namespace thermal {

// Enumeration of supported planetary bodies
enum class PlanetaryBody {
    MOON,
    MERCURY,
    EUROPA,
    ENCELADUS,
    TITAN,
    MARS,
    CERES,
    CUSTOM
};

// Structure to hold all thermophysical properties
struct ThermophysicalProperties {
    // Orbital and rotational parameters
    double day;           // Rotation period (seconds)
    double year;          // Orbital period (seconds)
    double rAU;           // Distance from Sun (AU)
    double S;             // Solar constant at orbit (W/m²)
    
    // Surface properties
    double albedo;        // Bond albedo
    double emissivity;    // Thermal emissivity
    double Qb;            // Internal heat flux (W/m²)
    
    // Thermal properties
    double ks;            // Surface thermal conductivity (W/m/K)
    double kd;            // Deep thermal conductivity (W/m/K)
    double rhos;          // Surface density (kg/m³)
    double rhod;          // Deep density (kg/m³)
    double H;             // Scale height for density/conductivity change (m)
    double cp0;           // Reference heat capacity (J/kg/K)
    
    // Variable property coefficients
    std::vector<double> cpCoeff;     // Heat capacity temperature coefficients
    std::vector<double> albedoCoef;  // Albedo variation coefficients

    // SPICE configuration
    std::string spice_metakernel;  // Meta-kernel filename for SPICE
};

// Default properties for supported planetary bodies
class PlanetaryData {
public:
    // Get properties for a specific body
    static ThermophysicalProperties getProperties(PlanetaryBody body) {
        switch (body) {
            case PlanetaryBody::EUROPA:
                return getEuropaProperties();
            case PlanetaryBody::MOON:
                return getMoonProperties();
            case PlanetaryBody::MERCURY:
                return getMercuryProperties();
            // Add other bodies as needed
            default:
                return getEuropaProperties(); // Default to Europa
        }
    }
    
    // Get properties by name
    static ThermophysicalProperties getPropertiesByName(const std::string& name) {
        // Convert to lowercase for case-insensitive comparison
        std::string lower_name = name;
        for (auto& c : lower_name) c = std::tolower(c);
        
        if (lower_name == "europa") return getProperties(PlanetaryBody::EUROPA);
        if (lower_name == "moon") return getProperties(PlanetaryBody::MOON);
        if (lower_name == "mercury") return getProperties(PlanetaryBody::MERCURY);
        // Add other bodies as needed
        
        // Default to Europa if name not recognized
        return getProperties(PlanetaryBody::EUROPA);
    }

private:
    // Europa properties
    static ThermophysicalProperties getEuropaProperties() {
        ThermophysicalProperties props;
        
        // Orbital parameters
        props.day = 3.55 * 24 * 3600;    // Europa's day in seconds (synchronous)
        props.year = 3.55 * 24 * 3600;   // Same as day for synchronous rotation
        props.rAU = 5.2;                 // Distance from Sun in AU
        props.S = 52.0;                  // Solar constant at Europa's orbit
        
        // Surface properties
        props.albedo = 0.5;              // Bond albedo
        props.emissivity = 0.9;          // Thermal emissivity
        props.Qb = 0.03;                 // Internal heat flux (W/m²)
        
        // Thermal properties
        props.ks = 0.002;                // Surface thermal conductivity (W/m/K)
        props.kd = 0.01;                 // Deep thermal conductivity (W/m/K)
        props.rhos = 100.0;              // Surface density (kg/m³)
        props.rhod = 450.0;              // Deep density (kg/m³)
        props.H = 0.07;                  // Scale height (m)
        props.cp0 = 900.0;               // Reference heat capacity (J/kg/K)
        
        // Variable property coefficients
        props.cpCoeff = {90.0, 7.49};    // Heat capacity temperature coefficients
        props.albedoCoef = {0.1, 0.05};  // Albedo variation coefficients
        
        // SPICE configuration
        props.spice_metakernel = "europa.tm";

        return props;
    }
    
    // Moon properties
    static ThermophysicalProperties getMoonProperties() {
        ThermophysicalProperties props;
        
        // Orbital parameters
        props.day = 29.5 * 24 * 3600;    // Lunar day in seconds
        props.year = 365.25 * 24 * 3600; // Earth year in seconds
        props.rAU = 1.0;                 // Same distance as Earth
        props.S = 1361.0;                // Solar constant at 1 AU
        
        // Surface properties
        props.albedo = 0.12;             // Bond albedo
        props.emissivity = 0.95;         // Thermal emissivity
        props.Qb = 0.02;                 // Internal heat flux (W/m²)
        
        // Thermal properties
        props.ks = 0.001;                // Surface thermal conductivity (W/m/K)
        props.kd = 0.01;                 // Deep thermal conductivity (W/m/K)
        props.rhos = 1300.0;             // Surface density (kg/m³)
        props.rhod = 1800.0;             // Deep density (kg/m³)
        props.H = 0.06;                  // Scale height (m)
        props.cp0 = 600.0;               // Reference heat capacity (J/kg/K)
        
        // Variable property coefficients
        props.cpCoeff = {0.75, 0.0};     // Heat capacity temperature coefficients
        props.albedoCoef = {0.05, 0.0};  // Albedo variation coefficients

        // SPICE configuration
        props.spice_metakernel = "moon.tm";
        
        return props;
    }
    
    // Mercury properties
    static ThermophysicalProperties getMercuryProperties() {
        ThermophysicalProperties props;
        
        // Orbital parameters
        props.day = 58.646 * 24 * 3600;  // Mercury day in seconds
        props.year = 87.969 * 24 * 3600; // Mercury year in seconds
        props.rAU = 0.387;               // Average distance from Sun in AU
        props.S = 9126.6;                // Solar constant at Mercury's orbit
        
        // Surface properties
        props.albedo = 0.07;             // Bond albedo
        props.emissivity = 0.93;         // Thermal emissivity
        props.Qb = 0.02;                 // Internal heat flux (W/m²)
        
        // Thermal properties
        props.ks = 0.002;                // Surface thermal conductivity (W/m/K)
        props.kd = 2.0;                  // Deep thermal conductivity (W/m/K)
        props.rhos = 1800.0;             // Surface density (kg/m³)
        props.rhod = 2200.0;             // Deep density (kg/m³)
        props.H = 0.1;                   // Scale height (m)
        props.cp0 = 800.0;               // Reference heat capacity (J/kg/K)
        
        // Variable property coefficients
        props.cpCoeff = {0.25, 0.0};     // Heat capacity temperature coefficients
        props.albedoCoef = {0.02, 0.0};  // Albedo variation coefficients

        // SPICE configuration
        props.spice_metakernel = "mercury.tm";
        
        return props;
    }
};

} // namespace thermal

#endif // PLANETARY_PROPERTIES_H