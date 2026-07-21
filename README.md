```
    ╔═══════════════════════════════════════════════════════════════╗
    ║    *      .    *     \   ||    //      * .   .      *         ║
    ║       .      *     \  \  ||   //    //       .             .  ║
    ║  *      .       \   \  \  ☀  //    //      .      *           ║
    ║     .         *  \   ...-'''''-...//    *         .           ║
    ║           _ ... .-'''''   PTM3D     ``''-.....,__      .      ║
    ║   __,.-''    Planetary Thermophysical Model 3D     `'-..., _  ║
    ║  ═══════════════════════════════════════════════════════════  ║
    ║      Coupled Thermal-Illumination for Planetary Bodies        ║
    ╚═══════════════════════════════════════════════════════════════╝
```

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/std/the-standard)

**PTM3D** is a high-performance computational framework for simulating surface and subsurface temperatures on planetary bodies. It combines ray-traced illumination modeling with SPICE ephemeris data and advanced thermophysical calculations to predict accurate thermal regimes on the Moon, Mercury, Europa, and other solar system objects.

## Quickstart

PTM3D's portable core is **CPU-first** (threaded); the OpenCL GPU path is an optional accelerator.
A 16 GB laptop runs real science on 1k–100k-face meshes.

```bash
# Build the portable CPU core (see INSTALL.md for dependency setup incl. CSPICE):
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPTM_ENABLE_GPU=OFF \
      -DPTM_ARCH_NATIVE=OFF -DCSPICE_ROOT="$CSPICE_ROOT"
cmake --build build --target ptm_thermal_sim -j
# Verify (analytic radiation gate):
cmake --build build --target test_bowl_crater -j && ./build/bin/test_bowl_crater
# Estimate a run's memory before launching:
python3 scripts/ptm_mem_estimate.py --faces 100000 --steps 360
```

- **Full install (Linux/macOS, per-platform deps):** [`INSTALL.md`](INSTALL.md)
- **10-minute laptop walkthrough:** [`TUTORIAL.md`](TUTORIAL.md)
- **Portability status / known caveats:** [`docs/PORTABILITY-AUDIT.md`](docs/PORTABILITY-AUDIT.md)
- **GPU build (OpenCL hosts):** add `-DPTM_ENABLE_GPU=ON`, then profile the host once with
  `./build/bin/ptm_thermal_sim --calibrate-gpu`.

## Key Features

- **Multi-body support**: Moon, Mercury, Europa, Enceladus, Titan, Mars, Ceres, plus a generic body interface for arbitrary objects with Keplerian or SPICE/Horizons ephemerides
- **Ray-traced illumination**: Intel Embree 4 with SIMD ray packets (AVX-512/AVX2), finite solar disk integration, eclipse geometry, and primary-body IR and reflected flux
- **SPICE ephemeris integration**: Accurate planetary positions and orientations via NASA NAIF kernels, with JPL Horizons fallback for small bodies
- **1D thermal conduction**: Crank-Nicolson implicit solver with temperature-dependent thermal conductivity and heat capacity, adaptive depth grids, and FFT-based spectral equilibration
- **Radiation exchange**: Inter-face thermal radiation with sparse or H-matrix compressed view factors, Jacobi/Gauss-Seidel radiosity solver, and adaptive thermal-radiation coupling iterations
- **GPU acceleration**: OpenCL backend for thermal solver, radiosity, and BVH ray tracing (macOS/Linux); HIP/CUDA support for thermal solver
- **Exosphere modeling**: Water vapor sublimation, ion sputtering, ice crystallization kinetics, composition-albedo feedback, and multi-timescale evolution for icy bodies
- **Ice stability depth**: Newton-Raphson solver for subsurface ice table location via vapor diffusion
- **Viewshed analysis**: Line-of-sight visibility calculations for surface mission planning
- **Flexible mesh input**: PLY, OBJ, VTK formats; built-in icosphere generator; DEM-to-mesh conversion (with GDAL); synthetic crater mesh generator
- **ParaView-compatible output**: VTK/PVD time series, binary statistics, CSV location time series, NetCDF and HDF5 export

## Applications

- Lunar permanently shadowed region (PSR) thermal mapping
- Water ice stability on Europa and icy moons
- Mercury extreme temperature variations
- Crater cold-trap volatile retention
- Surface mission thermal environment planning
- Ice shell thickness and polar wander modeling (Europa)
- Exosphere source rate estimation from thermal models

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Simulation Workflow](#simulation-workflow)
- [Architecture](#architecture)
- [Command-Line Reference](#command-line-reference)
- [Configuration File](#configuration-file)
- [Output Formats](#output-formats)
- [Graphical User Interface](#graphical-user-interface)
- [Python Tools](#python-tools)
- [Examples](#examples)
- [Validation](#validation)
- [Documentation](#documentation)
- [Citation](#citation)
- [License](#license)

---

## Installation

### Prerequisites

| Dependency | Version | Required | Purpose |
|------------|---------|----------|---------|
| CMake | 3.10+ | Yes | Build system |
| C++ Compiler | C++17 | Yes | GCC 8+, Clang 10+, or Apple Clang |
| Embree | 4.0+ | Yes | Ray tracing for illumination and visibility |
| CFITSIO | Any | Yes | FITS file I/O for albedo and property maps |
| Eigen3 | 3.0+ | Yes | Linear algebra (H-matrix SVD, radiosity) |
| CSPICE | Latest | Yes | NASA NAIF ephemeris toolkit |
| OpenMP | Built-in | Yes | Thread-level parallelization |
| yaml-cpp | 0.6+ | Recommended | YAML configuration file parsing |
| VTK | 8.0+ | Optional | Mesh I/O (PLY/OBJ/VTK import/export) |
| OpenCL | 1.2+ | Optional | GPU acceleration |
| GDAL | 3.0+ | Optional | DEM-to-mesh conversion |

### macOS

```bash
# Install Homebrew dependencies
brew install cmake libomp eigen embree cfitsio yaml-cpp

# Optional
brew install vtk gdal

# Download CSPICE from https://naif.jpl.nasa.gov/naif/toolkit_C.html
# Extract to ~/software/cspice/
```

### Linux (Ubuntu/Debian)

```bash
# Core dependencies
sudo apt-get install -y cmake build-essential libomp-dev libeigen3-dev \
    libcfitsio-dev libyaml-cpp-dev

# Embree (may need to build from source on older distros)
sudo apt-get install -y libembree-dev

# Optional
sudo apt-get install -y libvtk9-dev libgdal-dev

# Download CSPICE from https://naif.jpl.nasa.gov/naif/toolkit_C.html
```

### Building from Source

```bash
git clone https://github.com/phayne/ptm3d.git
cd ptm3d
mkdir -p build && cd build

# Configure (adjust CSPICE_ROOT as needed)
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCSPICE_ROOT=/path/to/cspice

# Build
cmake --build . -j$(nproc)

# Executables are in build/bin/
ls bin/
# ptm_thermal_sim  ptm_exosphere_main  albedo_map_generator
# crater_mesh_generator  dark_horse_viewshed  map_processor  ...
```

### Build Options

| CMake Flag | Default | Description |
|------------|---------|-------------|
| `ENABLE_GPU` | OFF | Enable GPU acceleration |
| `GPU_BACKEND` | AUTO | GPU backend: `AUTO`, `OPENCL`, `HIP`, `CUDA` |
| `ENABLE_AVX2` | ON | AVX2 SIMD optimization (x86-64) |
| `ENABLE_AVX512` | ON | AVX-512 SIMD optimization (x86-64) |
| `BUILD_TESTS` | OFF | Build test executables |
| `BUILD_PYTHON_BINDINGS` | OFF | Build pybind11 Python module |
| `PTM_USE_VTK_READER` | ON | VTK mesh I/O support |

**GPU Support (Optional)**:
```bash
# Auto-detect available backend (HIP > CUDA > OpenCL)
cmake .. -DENABLE_GPU=ON -DGPU_BACKEND=AUTO

# Force OpenCL (recommended: full feature support on macOS and Linux)
cmake .. -DENABLE_GPU=ON -DGPU_BACKEND=OPENCL
```

OpenCL is the primary GPU backend with support for the thermal solver, radiosity, and BVH ray tracing. HIP/CUDA backends support the thermal solver only.

---

## Quick Start

### 1. Minimal Test

```bash
cd build
./bin/ptm_thermal_sim \
    --body Moon \
    --resolution 4 \
    --duration 0.5 \
    --timesteps 24 \
    --thermal-mesh \
    --output ../output/quick_test
```

### 2. Using a Configuration File

```bash
./bin/ptm_thermal_sim --config ../examples/quick_test.yaml
```

### 3. Full Europa Simulation

Europa is tidally locked to Jupiter (orbital period = 3.55 days), receives Jupiter IR and reflected sunlight, and has ice surface properties calibrated to thermal inertia ~60 SI units.

```bash
./bin/ptm_thermal_sim \
    --body Europa \
    --resolution 5 \
    --duration 3.55 \
    --timesteps 144 \
    --thermal-mesh \
    --thermal-threads 8 \
    --equilibration-orbits 10 \
    --output ../output/europa_full \
    --verbose
```

### 4. Crater with Radiation Exchange

```bash
# Generate a bowl crater mesh
./bin/crater_mesh_generator --diameter 10 --depth 1.25 \
    --output data/meshes/crater.ply

# Run simulation with inter-face radiation
./bin/ptm_thermal_sim \
    --mesh-file data/meshes/crater.ply \
    --body Moon \
    --duration 29.5 \
    --timesteps 354 \
    --thermal-mesh \
    --radiation-exchange \
    --rad-hmatrix \
    --output output/crater_sim
```

### 5. Visualize Results

```bash
# Convert binary output to ParaView format
python src/python/vtk_converter.py output/europa_full output/paraview

# Open in ParaView: File > Open > output/paraview/statistics.pvd
```

![Example Output](docs/figures/mons_mouton_temperature.gif)
*Surface temperature evolution on Mons Mouton (lunar south polar region) over one lunation.*

---

## Simulation Workflow

```
                              ┌─────────────────┐
                              │     INPUTS       │
                              └────────┬─────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
              v                        v                        v
     ┌────────────────┐      ┌────────────────┐      ┌────────────────┐
     │  Mesh File     │      │ SPICE Kernels  │      │  Config File   │
     │ (PLY/OBJ/VTK)  │      │  (Ephemeris)   │      │    (YAML)      │
     └───────┬────────┘      └───────┬────────┘      └───────┬────────┘
             │                       │                        │
             └───────────────────────┼────────────────────────┘
                                     │
                                     v
                        ┌────────────────────────┐
                        │    MESH GENERATION      │
                        │  Icosphere / DEM import │
                        │  Normals, areas, coords │
                        └───────────┬─────────────┘
                                    │
                                    v
                        ┌────────────────────────┐
                        │   ILLUMINATION MODEL    │
                        │  Ray tracing (Embree)   │
                        │  Direct solar flux      │
                        │  Primary reflection     │
                        │  Primary IR emission    │
                        │  Eclipse computation    │
                        └───────────┬─────────────┘
                                    │
                                    v
                        ┌────────────────────────┐
                        │     THERMAL MODEL       │
                        │  Spectral equilibration │
                        │  Crank-Nicolson solver  │
                        │  T-dependent k(T), cp(T)│
                        └───────────┬─────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    v                               v
        ┌───────────────────┐           ┌───────────────────┐
        │ RADIATION EXCHANGE│           │   (no coupling)   │
        │  View factors     │           └─────────┬─────────┘
        │  H-matrix / CSR   │                     │
        │  Radiosity solver │                     │
        └─────────┬─────────┘                     │
                  │                               │
                  v                               │
        ┌───────────────────┐                     │
        │  COUPLING LOOP    │                     │
        │  T -> B -> Q_ind  │                     │
        │  Selective re-run │                     │
        │  Convergence check│                     │
        └────┬────┬─────────┘                     │
             │    │                               │
             │    └── iterate ──> back to         │
             │        Thermal Model               │
             │                                    │
             └────────────────┬───────────────────┘
                              │
                              v
                   ┌───────────────────┐
                   │  ICE STABILITY    │
                   │  Vapor diffusion  │
                   │  Newton-Raphson   │
                   │  Ice table depth  │
                   └────────┬──────────┘
                            │
                            v
                   ┌────────────────────┐
                   │      OUTPUTS       │
                   └────────┬───────────┘
                            │
           ┌────────────────┼────────────────┐
           │                │                │
           v                v                v
  ┌────────────────┐ ┌──────────────┐ ┌──────────────┐
  │  Binary Files  │ │   VTK/PVD    │ │ CSV Series   │
  │  temperature   │ │  ParaView    │ │ Locations    │
  │  illumination  │ │  compatible  │ │ Time series  │
  │  statistics    │ │  animation   │ │ Depth data   │
  └────────────────┘ └──────────────┘ └──────────────┘
```

---

## Architecture

### Source Layout

```
src/cpp/
├── main/             Main driver (ptm_thermal_sim.cpp)
├── thermal/          1D heat conduction, spectral equilibration
├── illumination/     Ray-traced solar flux with SPICE ephemeris
├── radiation/        View factors, H-matrix compression, radiosity
├── gpu/              OpenCL/HIP/CUDA acceleration
├── mesh/             Mesh I/O and geometry utilities
├── orbital/          Kepler propagation, body-fixed frames
├── maps/             Albedo/property FITS maps, spatial interpolation
├── constants/        Physical and planetary constants
├── ice/              Ice stability depth solver
├── tools/            Config parser, crater/DEM mesh generators
├── fileio/           Memory-mapped binary I/O
├── exosphere/        Sublimation, sputtering, ice evolution
└── viewshed/         Line-of-sight visibility analysis
```

### Core Modules

| Module | Key Classes | Description |
|--------|-------------|-------------|
| **thermal** | `thermal::Model`, `thermal::Profile`, `thermal::Config` | 1D Crank-Nicolson heat conduction with temperature-dependent properties, adaptive depth grids, and FFT-based spectral equilibration for fast spin-up |
| **illumination** | `IlluminationModel` | Shadow ray tracing (Embree), finite solar disk integration, primary-body reflected and thermal IR flux, eclipse geometry via SPICE |
| **radiation** | `RadiationExchangeManager`, `RadiositySolver`, `ViewFactorMatrix`, `HMatrixViewFactor` | Sparse or H-matrix compressed view factors, Jacobi/Gauss-Seidel radiosity iteration, adaptive thermal-radiation coupling |
| **gpu** | `ThermalOpenCLSolver`, `RadiosityOpenCLSolver`, `BVHRayTracer` | GPU-accelerated thermal solver (register-based Thomas algorithm), radiosity (SpMV Jacobi), and BVH shadow ray tracing |
| **orbital** | `GenericBody`, `KeplerPropagator` | Keplerian orbit propagation, SPICE/Horizons ephemeris, body-fixed coordinate frames, spin state |
| **mesh** | `ptm::Mesh` | Icosphere generation, PLY/OBJ/VTK import, face normals/areas/centers, VTK export |
| **ice** | Ice stability solver | Vapor diffusion and Newton-Raphson ice table depth |

### Research Modules

| Module | Key Classes | Description |
|--------|-------------|-------------|
| **exosphere** | `ExospherePhysics`, `SputteringModel`, `IceLayerModel`, `IceCrystallizationModel` | Water vapor sublimation rates, ion sputtering yields, amorphous-to-crystalline ice phase transitions, regolith lag formation, composition-albedo feedback, multi-timescale epoch evolution |
| **viewshed** | Viewshed analyzer | Line-of-sight visibility from surface waypoints, terrain occlusion, mission traverse planning |

### Physics Highlights

**Thermal Model**:
- Temperature-dependent thermal conductivity: $k(T) = k_c \cdot (1 + \chi \cdot (T/350)^3)$
- Heat capacity: polynomial for lunar regolith; linear $c_p(T) = 7.49 \cdot T$ for water ice (Europa)
- Adaptive depth grid with logarithmic spacing
- Spectral equilibration: FFT decomposition of the diurnal flux into Fourier harmonics, each solved analytically with skin-depth scaling $\delta_n = \delta_1/\sqrt{n}$, converging in 3-5 iterations vs. thousands of time-stepping cycles
- Validated against Lunar Diviner data (RMS error <4 K)

**Radiation Exchange**:
- Radiosity equation: $B_i = \varepsilon_i \sigma T_i^4 + \rho_i \sum_j F_{ij} B_j + \rho_i Q_{\text{solar},i}$
- View factor methods: centroid (far field) and subdivision (near field), selected adaptively by dimensionless separation $N_s = r/\sqrt{A}$
- H-matrix compression with octree spatial partitioning, ACA low-rank approximation, and $O(N \log N)$ matrix-vector products
- Adaptive thermal-radiation coupling: selective face re-run, warm-start solver, convergence on temperature and flux criteria

**Exosphere Model**:
- Sublimation-driven water vapor loss with Hertz-Knudsen kinetics
- Ion sputtering yields (magnetospheric bombardment)
- Amorphous-to-crystalline ice transition (Arrhenius kinetics)
- Composition-albedo feedback: non-ice lag concentration lowers albedo, increasing temperature (positive feedback for thermal segregation)
- Multi-timescale operator splitting: diurnal, seasonal, and geological epochs

![Thermal Wave Profiles](docs/figures/thermal_wave_profiles.png)
*Subsurface temperature profiles at different local solar times showing the diurnal thermal wave. Temperature evolution with depth and time; the white dashed line indicates the thermal skin depth (~3.4 cm).*

---

## Command-Line Reference

### Simulation Setup

```
--config <file>               Load YAML configuration file
--body, -b <name>             Body name (Moon, Mercury, Europa, Mars, Ceres, ...)
--start <time>                Start time (ISO 8601, default: 2025-01-01T00:00:00)
--resolution <n>              Global icosphere resolution (default: 5)
--duration <days>             Simulation duration in days
--timesteps <n>               Number of output timesteps
--output <dir>                Output directory
--verbose                     Enable detailed logging
```

### Mesh

```
--mesh-file <file>            Import mesh (PLY, OBJ, VTK)
--mesh-scale <factor>         Scale imported mesh (default: 1.0)
--export-basic-mesh           Export basic mesh to VTK
```

### Thermal

```
--thermal-mesh                Run thermal model for all faces
--thermal-threads <n>         Number of parallel threads (0 = auto)
--equilibration-orbits <n>    Diurnal cycles for equilibration (default: 1)
--use-gpu-thermal             Use GPU thermal solver (OpenCL)
--geothermal-heat-flow <Q>    Bottom boundary heat flow (W/m^2)
--thermal-m <n>               Layers in upper skin depth (default: 10)
--thermal-n <n>               Layer growth factor (default: 15)
--thermal-b <n>               Skin depths to bottom (default: 8)
```

### Radiation Exchange

```
--radiation-exchange          Enable inter-face radiation exchange
--rad-hmatrix                 Use H-matrix compression (recommended for >50K faces)
--rad-use-gpu                 Use GPU radiosity solver (OpenCL)
--rad-view-factor-file <f>    Save/load pre-computed view factors
--rad-coupling-iter <n>       Max coupling iterations (default: 10)
--rad-coupling-temp-tol <K>   Temperature convergence threshold (default: 2.0 K)
--rad-coupling-flux-tol <W>   Flux change threshold (default: 0.1 W/m^2)
--rad-coupling-face-frac <f>  Face fraction still changing (default: 0.01)
--rad-aca-tolerance <t>       ACA compression tolerance (default: 1e-3)
--rad-hmatrix-threads <n>     H-matrix build threads (0 = auto)
```

### Illumination

```
--albedo-map <file>           Custom albedo FITS map
--no-primary-reflection       Disable reflected light from primary body
--shadow-flux                 Enable shadow/crater scattered flux
--positions <file>            Export body positions to CSV
```

### Output

```
--output-format <fmt>         Output format: "binary" or "vtk" (default: binary)
--vtk-mode <mode>             VTK mode: "thermal", "complete", "both"
--include-primary-ir          Include primary IR in VTK output
--locations <file>            Extract illumination for lat/lon locations
--depths <file>               Export subsurface temperature profiles
--series-output <file>        Time series output file
```

Run `./bin/ptm_thermal_sim --help` for the complete reference.

---

## Configuration File

PTM3D accepts YAML configuration files via `--config`. Command-line flags override config file values. The top-level sections are:

```yaml
simulation:
  body: "Europa"
  start_time: "2025-01-01T00:00:00"
  duration_days: 3.55
  timesteps: 144

mesh:
  resolution: 5
  import_file: ""           # PLY/OBJ/VTK path (overrides icosphere)
  use_topographic: false    # Use topographic DEM

thermal:
  run_thermal_mesh: true
  threads: 0                # 0 = auto-detect
  equilibration_orbits: 10
  geothermal_heat_flow: -1  # -1 = planet default

illumination:
  include_primary_reflection: true
  use_batch_processing: true
  albedo_maps_directory: "data/albedo_maps"

radiation:
  enabled: false
  use_hmatrix: false
  max_coupling_iterations: 10
  coupling_temp_threshold: 2.0   # K
  coupling_flux_threshold: 0.1   # W/m^2

output:
  directory: "output/simulation"
  format: "binary"

generic_body:                # For arbitrary bodies
  name: "MyAsteroid"
  radius_km: 50.0
  semi_major_axis_au: 2.5
  spin_period_hours: 6.0
  albedo: 0.1
  emissivity: 0.95
  # ... (see docs/configuration.md for full reference)
```

See [`examples/`](examples/) for 30 example configurations covering Europa, Moon, Mercury, Vesta, and custom bodies.

---

## Output Formats

PTM3D generates several output types:

```
output/
├── mesh.ply                    # Mesh geometry
├── statistics/
│   ├── temperature_avg.bin     # Mean temperature per face
│   ├── temperature_max.bin     # Maximum temperature
│   ├── temperature_min.bin     # Minimum temperature
│   ├── illumination_avg.bin    # Mean solar flux
│   └── ice_depth.bin           # Ice stability depth
├── temperature/
│   ├── temperature_00001.bin   # Per-timestep temperatures
│   └── ...
└── illumination_series.csv     # Location time series (if --locations)
```

**Binary format**: Single-precision float arrays, one value per mesh face.

**VTK/PVD export**: Use `python src/python/vtk_converter.py` to create ParaView-compatible time series.

**Additional exports**: NetCDF (`netcdf_converter.py`), HDF5 (`ptm_hdf5.py`), GeoTIFF (`ptm_geotiff.py`).

---

## Graphical User Interface

PTM3D includes three PyQt6-based GUIs for configuring simulations and visualizing results:

| GUI | Script | Purpose |
|-----|--------|---------|
| **PTM3D Main** | `gui/run_gui.py` | Thermal simulation configuration and 3D visualization |
| **Exosphere** | `gui/exosphere_gui.py` | Exosphere sublimation/sputtering modeling |
| **Viewshed** | `gui/viewshed_gui.py` | Line-of-sight visibility analysis |

### Quick Start

```bash
cd gui/
pip install -r requirements.txt
python run_gui.py
```

### Features

- **Tabbed configuration**: Simulation, mesh, thermal, radiation, illumination, and output settings
- **3D visualization**: Interactive PyVista-based mesh rendering with temperature colormap
- **2D analysis**: Histograms, latitude plots, and time series
- **Live preview**: Real-time YAML configuration preview
- **Console output**: Color-coded simulation progress and logging

See [gui/README.md](gui/README.md) for detailed GUI documentation.

---

## Python Tools

### Post-Processing Scripts (`src/python/`)

| Script | Purpose |
|--------|---------|
| `vtk_converter.py` | Convert binary output to VTK/PVD for ParaView |
| `vtk_converter_exosphere.py` | Convert exosphere output to VTK |
| `netcdf_converter.py` | Export to NetCDF |
| `ptm_hdf5.py` | Export to HDF5 |
| `ptm_geotiff.py` | Export to GeoTIFF (geospatial) |
| `plot_depth_profiles.py` | Subsurface temperature-depth profiles |
| `extract_timeseries.py` | Extract data for specific lat/lon locations |
| `lunar_validation_figure.py` | Validation plots vs. Hayne et al. (2017) |
| `europa_visualization.py` | Europa-specific visualization |
| `exosphere_analysis.py` | Exosphere output analysis |
| `thermal_vis.py` | Thermal field visualization |
| `viewshed_pv_animator.py` | Animated viewshed in ParaView |
| `ptm_paraview_converter.py` | ParaView format conversion |

### Analysis Scripts (`scripts/`)

Research analysis scripts for ice shell modeling, polar wander, ALMA data comparison, and seasonal thermal surveys.

### Python Packages (`src/python/`)

| Package | Purpose |
|---------|---------|
| `ojakangas_stevenson/` | Equilibrium ice shell thickness model (O&S 1989) |
| `ptm_ice/` | Ice shell thermal analysis |
| `horizons/` | JPL Horizons ephemeris API wrapper |
| `damit/` | DAMIT asteroid shape model database |

### Example: Convert and Visualize

```bash
# Convert to ParaView format
python src/python/vtk_converter.py output/simulation output/paraview

# Export to NetCDF
python src/python/netcdf_converter.py \
    --input output/paraview/statistics.pvd \
    --output output/statistics.nc \
    --format pvd
```

---

## Examples

The [`examples/`](examples/) directory contains 30 configuration files:

| Category | Examples | Description |
|----------|----------|-------------|
| **Europa** | `quick_test.yaml`, `europa_high_res.yaml` | Global thermal models at various resolutions |
| **Moon (global)** | `moon_global_ico6.yaml`, `moon_global_ico7.yaml` | Full lunation global models |
| **Moon (regional)** | `moon_amundsen-seasonal1.yaml`, `leibniz_disco.yaml` | Polar crater seasonal analysis |
| **Moon (heat flow)** | `moon_heatflow_high.yaml`, `moon_heatflow_low.yaml` | Geothermal heat flow scenarios |
| **Moon (DEM)** | `haworth5m_heatflow_*.yaml` | 5-meter DEM crater simulations |
| **Validation** | `lunar_validation.yaml` | Comparison with Diviner observations |
| **Asteroids** | `vesta_horizons-test.yaml`, `justitia_*.yaml` | Small body thermal modeling |
| **Eclipse** | `moon-eclipse.yaml` | Lunar eclipse shadow flux |

### Extract Time Series for Specific Locations

```bash
# Create locations file (latitude longitude name)
cat > locations.txt << EOF
0.0 0.0 SubsolarPoint
45.0 0.0 MidLatitude
85.0 0.0 NearPole
EOF

./bin/ptm_thermal_sim \
    --config examples/lunar_validation.yaml \
    --locations locations.txt \
    --series-output illumination_series.csv
```

---

## Validation

PTM3D has been validated against:

- **Lunar Diviner** radiometer measurements (RMS <4 K)
- **Hayne et al. (2017)** thermophysical model and grid optimization methodology
- **Potter et al. (2023)** H-matrix view factor implementation
- **Ingersoll et al. (1992)** analytical bowl-crater radiation exchange (test case: `test_bowl_crater`)

![Validation](docs/figures/lunar_validation.png)
*PTM3D validation against Hayne et al. (2017) analytical model for the lunar equator. (a) Temperature vs. local solar time. (b) Maximum and minimum temperatures vs. latitude. PTM3D reproduces the expected diurnal cycle with noon temperatures of ~388 K and nighttime temperatures of ~99 K.*

---

## Documentation

Detailed documentation is in [`docs/`](docs/):

| Document | Description |
|----------|-------------|
| [Thermal Model](docs/thermal-model.md) | 1D heat conduction physics, Crank-Nicolson solver, spectral equilibration |
| [Illumination Model](docs/illumination-model.md) | Ray-traced solar flux, primary IR/reflection, eclipse geometry |
| [Radiation Exchange](docs/radiation-exchange.md) | View factors, H-matrix compression, radiosity solver, thermal coupling |
| [GPU Acceleration](docs/gpu-implementation-plan.md) | OpenCL/HIP/CUDA backend architecture and usage |
| [Configuration Guide](docs/configuration.md) | Complete YAML configuration reference |
| [API Reference](docs/api-reference.md) | C++ class and function documentation |

---

## Citation

If you use PTM3D in your research, please cite:

```bibtex
@software{ptm3d,
  author = {Hayne, Paul O.},
  title = {PTM3D: Planetary Thermophysical Model 3D},
  year = {2026},
  institution = {University of Colorado Boulder},
  url = {https://github.com/phayne/ptm3d}
}
```

### Heritage References

The thermal model is based on methods described in:

- Hayne, P. O., et al. (2017). Global regolith thermophysical properties of the Moon from the Diviner Lunar Radiometer Experiment. *J. Geophys. Res. Planets*, 122, 2371-2400. https://doi.org/10.1002/2017JE005387

- Hayne, P. O. & Aharonson, O. (2015). Thermal stability of ice on Ceres with rough topography. *J. Geophys. Res. Planets*, 120, 1567-1584. https://doi.org/10.1002/2015JE004887

- Vasavada, A. R., et al. (2012). Lunar equatorial surface temperatures and regolith properties from the Diviner Lunar Radiometer Experiment. *J. Geophys. Res. Planets*, 117, E00H18. https://doi.org/10.1029/2011JE003987

- Paige, D. A., et al. (2010). Diviner lunar radiometer observations of cold traps in the Moon's south polar region. *Science*, 330, 479-482. https://doi.org/10.1126/science.1187726

The H-matrix view factor compression is based on:

- Potter, S. F., et al. (2023). Fast hierarchical low-rank view factor matrices for thermal irradiance on planetary surfaces. *J. Comput. Phys.: X*, 17, 100130. https://doi.org/10.1016/j.jcpx.2023.100130

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [Sam Potter](https://github.com/sampotter) for H-matrix view factor algorithms and [fluxpy](https://github.com/sampotter/fluxpy)
- NASA NAIF for the SPICE toolkit and ephemeris data
- Intel for the Embree ray tracing library
- Lunar Reconnaissance Orbiter Diviner team for validation data
- NASA Lunar Vulkan Imaging and Spectroscopy Explorer (Lunar-VISE)
- NASA Europa Clipper mission
- NASA Precursor Science Investigations for Europa (PSIE)

---

**Questions?** Open an issue on GitHub or contact Paul Hayne (paul.hayne@colorado.edu).
