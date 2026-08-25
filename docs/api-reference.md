# API Reference

PTM3D is used primarily through `ptm_thermal_sim` and a YAML configuration; this
page is an orientation to the C++ types a caller linking against the libraries
will meet. Declarations are abridged — the headers named beside each type are
authoritative.

## Configuration

### `ptm::SimulationConfig` — `tools/config_parser.h`

The parsed configuration. Its nested structs mirror the YAML sections
one-for-one (`time`, `mesh`, `thermal`, `illumination`, `radiation`, `output`,
`generic_body`, `thermophysical`), and every field carries the default the
parser applies when the key is absent. Key names and defaults are tabulated in
[configuration.md](configuration.md).

```cpp
namespace ptm {
struct SimulationConfig {
    struct TimeParams   { std::string start_time; double duration_days; int timesteps; std::string body; } time;
    struct MeshParams   { int resolution; std::string import_file, occluder_file; double scale_factor; /* ... */ } mesh;
    struct ThermalParams{ bool run_thermal_mesh; int threads, grid_m, grid_n, grid_b; double equilibration_orbits; /* ... */ } thermal;
    struct IlluminationParams { bool include_primary_reflection, use_batch_processing, use_albedo_map; /* ... */ } illumination;
    struct RadiationParams    { bool enabled, band_separated; std::string coupling_mode, exchange_selection; /* ... */ } radiation;
    struct OutputParams       { std::string directory, format, vtk_mode; /* ... */ } output;
    /* ... */
};

class ConfigParser {
public:
    SimulationConfig loadFromYAML(const std::string& filename);
};
}
```

A YAML file that fails to parse is fatal: the loader reports `FATAL:` and the
program exits 1 rather than falling back to built-in defaults.

### `radiation::RadiationConfig` — `radiation/radiation_config.h`

The radiation module's own configuration, populated from
`SimulationConfig::radiation`. Field-by-field documentation, including which
defaults matter and why, is in [radiation-exchange.md §9](radiation-exchange.md).

```cpp
namespace radiation {
struct RadiationConfig {
    bool   enabled              = false;
    bool   band_separated       = true;   // visible + thermal-IR radiosity
    enum class ExchangeSelection { All, Shadow };
    ExchangeSelection exchange_selection = ExchangeSelection::All;
    double min_shadow_fraction  = 0.3;    // Shadow selection only
    bool   use_hmatrix          = false;
    bool   use_aca              = true;
    double aca_tolerance        = 1e-3;
    bool   use_monte_carlo_vf   = false;
    int    mc_rays_per_face     = 100000;
    int    max_iterations       = 100;    // radiosity sweeps
    double convergence_tolerance = 0.01;
    int    convergence_check_interval = 0;  // GPU only; 0 = auto, sized per matrix
    bool   use_gpu              = false;
    bool   beaming              = false;  // Smith beamed emission: THE SWITCH
    double rms_slope_deg        = 0.0;    // RMS sub-facet slope w [deg]: THE KNOB
    /* ... */
};
}
```

`coupling_mode`, `coupling_select_dt` and the other Picard-loop settings live on
`SimulationConfig::radiation` (as strings and doubles) and are consumed by the
coupling loop in `ptm_thermal_sim.cpp`, not by `radiation::RadiationConfig`.

## Mesh

### `ptm::Mesh`, `ptm::MeshUtils` — `mesh/mesh_utils.h`

```cpp
namespace ptm {
struct Mesh {
    std::vector<std::array<double, 3>> vertices;
    std::vector<std::array<int, 3>>    faces;
    std::vector<std::array<double, 3>> face_normals;
    std::vector<std::array<double, 3>> face_centers;
    std::vector<double> latitudes, longitudes, elevations;
    void        setProperty(const std::string& key, const std::string& value);
    std::string getProperty(const std::string& key, const std::string& fallback = "") const;
};

class MeshUtils {
public:
    void setMeshResolution(int resolution);
    void setupMesh();                                   // generate an icosphere
    void createEllipsoidMesh(int resolution, double a_km, double b_km, double c_km);
    bool loadMeshFromFile(const std::string& filename); // PLY / OBJ / VTK
    bool importTriangulatedMesh(const std::string& filename, double scale_factor = 1.0);
    bool exportMeshToVTK(const std::string& filename) const;
    void calculateFaceProperties(bool auto_orient_normals = true);
    std::shared_ptr<Mesh> getMesh() const;
    double                       getFaceArea(int face_idx) const;
    std::array<double, 3>        getFaceNormal(int face_idx) const;
    std::pair<double, double>    getFaceLatLon(int face_idx) const;
};
}
```

`getFaceArea` returns native mesh units. Icosphere meshes are generated in
kilometers and DEM meshes are imported in meters; the engine infers the unit at
startup and refuses to run when the inference is ambiguous. See
[KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md).

## Thermal solver

### `thermal::Planet`, `thermal::Profile`, `thermal::Model` — `thermal/thermal_model.h`

`Planet` holds the body's thermophysical and orbital properties; `Profile` owns
one face's depth grid and temperature column; `Model` drives a single column
through time.

```cpp
namespace thermal {
class Planet {
public:
    std::string name;
    double albedo, emissivity;
    double day, year, rAU, S, Qb;      // seconds, seconds, AU, W/m^2, W/m^2
    double ks, kd, rhos, rhod, H, cp0; // conductivity, density, scale height
    std::vector<double> cpCoeff;       // heat-capacity polynomial
    std::vector<double> albedoCoef;    // incidence-angle albedo polynomial
    double body_radius_km = 0.0;       // 0 = use the mesh as-is
    explicit Planet(const std::string& name = "Europa");
    explicit Planet(PlanetaryBody body);
};

class Profile { public: Planet planet; double emissivity, chi; int nlayers; /* ... */ };

class Model {
public:
    Planet  planet;
    Profile profile;
    double  lat, t, dt, equiltime, endtime;
    Eigen::MatrixXd flux_arr;   // absorbed surface flux time series
    Eigen::MatrixXd T;          // temperature output
    int N_steps, N_z;
};
}
```

### `thermal::PlanetaryData` — `thermal/planetary_properties.h`

Built-in property sets, by enum or by name:

```cpp
namespace thermal {
enum class PlanetaryBody { MOON, MERCURY, EUROPA, ENCELADUS, TITAN, MARS, CERES, CUSTOM };

class PlanetaryData {
public:
    static ThermophysicalProperties getProperties(PlanetaryBody body);
    static ThermophysicalProperties getPropertiesByName(const std::string& name);
};
}
```

Only `MOON`, `MERCURY` and `EUROPA` have property sets; other names fall back to
placeholder defaults, print a note on stderr, and expect a `generic_body:` block
to supply the real values.

## Illumination

### `IlluminationModel` — `illumination/illumination_model.h`

Computes direct solar flux, primary-body reflected flux and primary-body thermal
IR per face and timestep, with Embree shadow ray tracing, finite solar-disk
sampling and eclipse geometry.

```cpp
std::vector<std::array<double, 3>> getSunPositions() const;
std::array<double, 3>              getSunPosition(double et);
```

## Radiation exchange

### `radiation::RadiationExchangeManager`, `radiation::RadiositySolver` — `radiation/radiation_exchange.h`

The manager owns face classification, view-factor construction and the radiosity
solver; the solver returns the absorbed indirect flux for every face. The
physics and the solve sequence are documented in
[radiation-exchange.md](radiation-exchange.md).

```cpp
namespace radiation {
class RadiositySolver {
public:
    void setSolarAlbedo(const std::vector<double>& solar_albedo);  // required for band separation
    void enableTwoBand();
    bool isTwoBand() const;
    std::vector<double> solve(const std::vector<double>& temperatures,
                              const std::vector<double>& solar_flux);
    int    getLastIterations() const;
    double getLastResidual()   const;
    bool   didConverge()       const;
};
}
```

`solve()` returns $Q_{\text{indirect}}$ for **all** faces; the exchange-selection
mask is applied afterwards by the manager.

### `radiation::computeMonteCarlo` — `radiation/mc_view_factor.h`

```cpp
ViewFactorMatrix computeMonteCarlo(const std::shared_ptr<ptm::Mesh>& mesh,
                                   RTCDevice device, RTCScene scene,
                                   const RadiationConfig& config,
                                   int rays_per_face);
```

## Output formats

Binary statistics files are `[int32 n_faces][float32 x n_faces]`. A reader must
consume the leading `int32`; the reference reader is `scripts/ptm_qa.py`. The
per-face thermal-state checkpoint written by `thermal.final_state_file` is
`uint32 magic 0x50544D54 ('PTMT')`, `uint32 n_faces`, `uint32 n_layers`, then
`n_layers * n_faces` `float32` values in layer-major order.

Field-by-field descriptions of every output directory are in
[USER_GUIDE.md](USER_GUIDE.md) and
[tutorials/06_interpreting_outputs.md](tutorials/06_interpreting_outputs.md).
