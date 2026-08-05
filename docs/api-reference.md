# API Reference

This document provides reference documentation for PTM3D's C++ classes and functions.

## Core Classes

### thermal::Model

Main thermal simulation class.

```cpp
namespace thermal {
class Model {
public:
    Model(const Config& config, const Planet& planet);
    void step(double dt, double Q_surface);
    double getSurfaceTemperature() const;
    const Profile& getProfile() const;
};
}
```

### thermal::Profile

Stores depth grid and temperature profile.

```cpp
namespace thermal {
class Profile {
public:
    std::vector<double> z;      // Depth positions (m)
    std::vector<double> T;      // Temperature (K)
    std::vector<double> k;      // Conductivity (W/m/K)
    std::vector<double> cp;     // Heat capacity (J/kg/K)
    std::vector<double> kappa;  // Diffusivity (m²/s)
};
}
```

### thermal::Planet

Planetary thermophysical properties.

```cpp
namespace thermal {
class Planet {
public:
    double ks;      // Surface thermal conductivity
    double kd;      // Deep thermal conductivity
    double rho;     // Density
    double cp0;     // Reference heat capacity
    double albedo;  // Bond albedo
    double emiss;   // Thermal emissivity
    double Tday;    // Day length (s)
    double Tyear;   // Year length (s)
    double rAU;     // Heliocentric distance (AU)
};
}
```

### IlluminationModel

Solar flux computation.

```cpp
class IlluminationModel {
public:
    IlluminationModel(const Mesh& mesh, const std::string& spice_kernel);
    void computeIllumination(double et, std::vector<double>& flux);
    void setAlbedoMap(const SpatialMap& albedo);
};
```

### RadiationExchangeManager

High-level radiation exchange API.

```cpp
class RadiationExchangeManager {
public:
    RadiationExchangeManager(const RadiationConfig& config);
    void computeViewFactors(const Mesh& mesh);
    void solveRadiosity(const std::vector<double>& T,
                        std::vector<double>& indirect_flux);
    void saveViewFactors(const std::string& filename);
    void loadViewFactors(const std::string& filename);
};
```

## Configuration Structures

### SimulationConfig

Main configuration structure (from config_parser.h).

### RadiationConfig

Radiation exchange parameters.

```cpp
struct RadiationConfig {
    bool use_hmatrix = false;
    int max_coupling_iterations = 10;
    double coupling_temp_threshold = 0.5;
    double coupling_flux_threshold = 0.1;
    double coupling_face_fraction = 0.01;  // legacy coupling_mode only
    std::string coupling_mode = "bounded"; // bounded | fixed | legacy (PTM-89)
    double coupling_select_dt = 0.0;       // K, bounded/fixed only
    // PTM-98: which faces keep the flux the radiosity solve already computed.
    ExchangeSelection exchange_selection = ExchangeSelection::All;  // All | Shadow
    double min_shadow_fraction = 0.3;      // Shadow only (PTM-99: now plumbed)
    std::string view_factor_file;
    bool verbose = false;
};
```

`coupling_mode` controls whether the Picard coupling's face selection and exit
criteria are decomposition-independent; see `docs/radiation-exchange.md` §7.2.1.

`exchange_selection` controls which faces keep their computed indirect flux. The
radiosity solve always covers every face; this is a post-solve mask. `Shadow` (the
pre-PTM-98 behavior) measures each face's shadow fraction over the CURRENT window
— equilibration steps plus output steps — so splitting a run into chunks silently
zeroes the indirect flux on faces that happen to be lit for most of a chunk. See
`docs/radiation-exchange.md` §7.2.2.

## Utility Functions

### Mesh I/O

```cpp
Mesh loadMeshPLY(const std::string& filename);
Mesh loadMeshOBJ(const std::string& filename);
void saveMeshVTK(const Mesh& mesh, const std::string& filename);
```

### SPICE Wrappers

```cpp
double utc2et(const std::string& utc);  // Convert UTC to Ephemeris Time
void getSunPosition(double et, double pos[3]);
void getBodyPosition(double et, const std::string& body, double pos[3]);
```

## Building Documentation

Full API documentation can be generated using Doxygen:

```bash
cd docs
doxygen Doxyfile
```
