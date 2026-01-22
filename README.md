```
 ____  _____ __  __ _____ ____
|  _ \|_   _|  \/  |___ /|  _ \
| |_) | | | | |\/| | |_ \| | | |
|  __/  | | | |  | |___) | |_| |
|_|     |_| |_|  |_|____/|____/

Planetary Thermophysical Model 3D
```

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/std/the-standard)

**PTM3D** is a high-performance computational framework for simulating surface and subsurface temperatures on planetary bodies. It combines ray-traced illumination modeling with SPICE ephemeris data and advanced thermophysical calculations to predict accurate thermal regimes on the Moon, Mercury, Europa, and other solar system objects.

![PTM3D Workflow](docs/figures/ptm3d_workflow.png)
*Figure 1: PTM3D simulation workflow showing coupled illumination-thermal-radiation modeling.*

## Key Features

- **Multi-body support**: Moon, Mercury, Europa, and other bodies with calibrated thermophysical properties
- **Ray-traced illumination**: Intel Embree 4 with SIMD optimization (AVX-512/AVX2)
- **SPICE ephemeris integration**: Accurate planetary positions via NASA NAIF kernels
- **1D thermal conduction**: Crank-Nicolson solver with temperature-dependent properties
- **Radiation exchange**: View factors with H-matrix compression for large meshes
- **GPU acceleration**: OpenCL backend for thermal solver and radiosity (macOS/Linux)
- **Ice stability depth**: Newton-Raphson solver for subsurface ice table location
- **Flexible mesh input**: PLY, OBJ, VTK formats with resolution control
- **ParaView-compatible output**: VTK/PVD time series for visualization

## Applications

- Lunar Permanently Shadowed Regions (PSRs) thermal mapping
- Water ice stability modeling on Europa and icy moons
- Mercury extreme temperature variations
- Mission thermal environment planning
- Crater thermal regime analysis

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Command-Line Reference](#command-line-reference)
- [Output Formats](#output-formats)
- [Graphical User Interface](#graphical-user-interface)
- [Python Tools](#python-tools)
- [Examples](#examples)
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
| Embree | 4.0+ | Yes | Ray tracing |
| CFITSIO | Any | Yes | FITS file I/O |
| Eigen3 | 3.0+ | Yes | Linear algebra |
| CSPICE | Latest | Yes | NASA ephemeris |
| OpenMP | Built-in | Yes | Parallelization |
| yaml-cpp | 0.6+ | Optional | YAML config files |
| VTK | 8.0+ | Optional | Mesh I/O |
| OpenCL | 1.2+ | Optional | GPU acceleration |

### macOS

```bash
# Install Homebrew dependencies
brew install cmake libomp eigen embree cfitsio yaml-cpp

# Optional: VTK for mesh I/O
brew install vtk

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

# Optional: VTK
sudo apt-get install -y libvtk9-dev

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
# ptm_thermal_sim  albedo_map_generator  crater_mesh_generator  ...
```

**GPU Support (Optional)**:
```bash
# Enable GPU acceleration (auto-detects available backend)
cmake .. -DENABLE_GPU=ON -DGPU_BACKEND=AUTO

# Or specify backend explicitly
cmake .. -DENABLE_GPU=ON -DGPU_BACKEND=OPENCL  # Recommended: full feature support
```
OpenCL is the primary GPU backend with full support for thermal solver, radiosity, and ray tracing on macOS and Linux. HIP/CUDA backends have thermal solver support only. See [GPU Acceleration](docs/gpu-implementation-plan.md) for details.

---

## Quick Start

### 1. Minimal Test (~30 seconds)

```bash
cd ptm3d/build
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

Example `quick_test.yaml`:
```yaml
simulation:
  body: "Moon"
  start_time: "2025-01-01T00:00:00"
  duration_days: 0.5
  timesteps: 24

mesh:
  resolution: 4

thermal:
  run_thermal_mesh: true
  threads: 0  # Auto-detect
  equilibration_orbits: 5

output:
  directory: "output/quick_test"
  format: "binary"
```

### 3. Full Europa Simulation

Europa is tidally locked to Jupiter (day = orbital period = 3.55 days), receives Jupiter IR and reflected sunlight, and has ice surface properties calibrated to thermal inertia ~60 SI units.

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

For GPU acceleration (recommended for high-resolution meshes):
```bash
./bin/ptm_thermal_sim \
    --body Europa \
    --resolution 6 \
    --duration 3.55 \
    --timesteps 144 \
    --thermal-mesh \
    --use-gpu-thermal \
    --output ../output/europa_gpu
```

### 4. Visualize Results

```bash
# Convert binary output to ParaView format
python src/python/vtk_converter.py output/europa_full output/paraview

# Open in ParaView
# File → Open → output/paraview/statistics.pvd
```

![Example Output](docs/figures/mons_mouton_temperature.gif)
*Figure 2: Surface temperature evolution on Mons Mouton (lunar south polar region) over one lunation.*

---

## Architecture

PTM3D uses a modular architecture with coupled physics engines:

![Architecture](docs/figures/architecture.png)
*Figure 3: PTM3D module architecture and data flow.*

### Core Modules

| Module | Location | Description |
|--------|----------|-------------|
| **thermal** | `src/cpp/thermal/` | 1D heat conduction with Crank-Nicolson solver |
| **illumination** | `src/cpp/illumination/` | Ray-traced solar flux with SPICE ephemeris |
| **radiation** | `src/cpp/radiation/` | View factors, H-matrix compression, radiosity |
| **gpu** | `src/cpp/gpu/` | GPU acceleration (OpenCL) |
| **mesh** | `src/cpp/mesh/` | Mesh I/O and utilities |
| **ice** | `src/cpp/ice/` | Ice stability depth calculation |

### Research Modules

| Module | Location | Description |
|--------|----------|-------------|
| **exosphere** | `src/cpp/exosphere/` | Water vapor sublimation and sputtering |
| **viewshed** | `src/cpp/viewshed/` | Visibility analysis for mission planning |

### Physics Overview

**Thermal Model**:
- Temperature-dependent thermal conductivity: $k(T) = k_c \cdot \left(1 + \chi \cdot (T/350)^3\right)$
- Heat capacity: polynomial for lunar regolith, linear $c_p(T) = 7.49 \cdot T$ for water ice (Europa)
- Adaptive depth grid with logarithmic spacing
- Validated against Lunar Diviner data (RMS error <4K)

![Thermal Wave Profiles](docs/figures/thermal_wave_profiles.png)
*Figure 4: (a) Subsurface temperature profiles at different local solar times showing the diurnal thermal wave. (b) Temperature evolution with depth and time; the white dashed line indicates the thermal skin depth (~3.4 cm).*

**Radiation Exchange**:
- Sparse view factor matrix with CSR storage
- H-matrix hierarchical compression (O(N log N) storage)
- Adaptive thermal-radiation coupling iterations

---

## Command-Line Reference

### Essential Options

```
--body, -b <name>         Body name (Moon, Mercury, or Europa)
--config <file>           Load YAML configuration file
--resolution <n>          Global mesh resolution (default: 5)
--duration <days>         Simulation duration in days
--timesteps <n>           Number of output timesteps
--output <dir>            Output directory
--verbose                 Enable detailed logging
```

### Thermal Options

```
--thermal-mesh            Run thermal model for all faces
--thermal-threads <n>     Number of parallel threads (0 = auto)
--equilibration-orbits <n> Diurnal cycles for equilibration (default: 10)
--use-gpu-thermal         Use GPU thermal solver (requires OpenCL)
```

### Radiation Exchange

```
--radiation-exchange      Enable inter-face radiation
--rad-hmatrix            Use H-matrix compression (recommended for >50K faces)
--rad-view-factor-file <f> Save/load view factors
--rad-coupling-iter <n>   Max coupling iterations (default: 10)
--rad-coupling-temp-tol <K> Convergence threshold (default: 0.5 K)
```

### Illumination

```
--albedo-map <file>       Custom albedo FITS map
--no-primary-reflection   Disable reflected light from primary body
--positions <file>        Export body positions to CSV
```

Run `./bin/ptm_thermal_sim --help` for the complete reference.

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

**Binary format**: Single-precision float arrays, one value per face.

**VTK conversion**: Use `python src/python/vtk_converter.py` to create ParaView-compatible files.

**NetCDF export**: Use `python src/python/netcdf_converter.py` for cross-platform analysis.

---

## Graphical User Interface

PTM3D includes a modern PyQt6-based GUI for configuring simulations and visualizing results.

![PTM3D GUI](docs/figures/gui_screenshot.png)
*Figure: PTM3D GUI with 3D visualization and configuration panels.*

### Quick Start

```bash
cd gui/
pip install -r requirements.txt
python run_gui.py
```

### Features

- **Tabbed configuration**: Simulation, mesh, thermal, radiation, and output settings
- **3D visualization**: Interactive PyVista-based mesh rendering with temperature colormap
- **2D analysis**: Histograms, latitude plots, and time series
- **Live preview**: Real-time YAML configuration preview
- **Console output**: Color-coded simulation progress and logging

See [gui/README.md](gui/README.md) for detailed GUI documentation.

---

## Python Tools

| Script | Purpose |
|--------|---------|
| `vtk_converter.py` | Convert binary output to VTK/PVD format |
| `netcdf_converter.py` | Export to NetCDF for cross-platform analysis |
| `plot_depth_profiles.py` | Visualize temperature vs depth from CSV output |
| `plot_thermal_debug.py` | Plot diurnal thermal wave profiles from debug output |
| `extract_timeseries.py` | Extract data for specific lat/lon locations |
| `lunar_validation_figure.py` | Generate validation plots comparing to Hayne et al. (2017) |

### Example: Convert and Visualize

```bash
# Convert to ParaView format
python src/python/vtk_converter.py output/simulation output/paraview

# Generate depth profile plots
python src/python/plot_depth_profiles.py output/simulation/thermal_debug.txt

# Export to NetCDF
python src/python/netcdf_converter.py \
    --input output/paraview/statistics.pvd \
    --output output/statistics.nc \
    --format pvd
```

---

## Examples

### Crater Thermal Analysis

```bash
# Generate a bowl crater mesh
./bin/crater_mesh_generator --diameter 10 --depth 1.25 \
    --output data/meshes/crater.ply

# Run simulation with radiation exchange
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

### Extract Time Series for Specific Locations

```bash
# Create locations file
cat > locations.txt << EOF
0.0 0.0 SubsolarPoint
45.0 0.0 MidLatitude
85.0 0.0 NearPole
EOF

./bin/ptm_thermal_sim \
    --config examples/quick_test.yaml \
    --locations locations.txt \
    --series-output illumination_series.csv
```

---

## Validation

PTM3D has been validated against:

- **Lunar Diviner** radiometer measurements (RMS <4K)
- **Hayne et al. (2017)** grid optimization methodology
- **Potter et al. (2023)** H-matrix implementation

![Validation](docs/figures/lunar_validation.png)
*Figure 5: PTM3D validation against Hayne et al. (2017) analytical model for the lunar equator. (a) Temperature vs. local solar time. (b) Maximum and minimum temperatures vs. latitude. PTM3D reproduces the expected diurnal cycle with noon temperatures of ~388 K and nighttime temperatures of ~99 K.*

---

## Documentation

For more detailed information, see the documentation in [`docs/`](docs/):

| Document | Description |
|----------|-------------|
| [Thermal Model](docs/thermal-model.md) | Physics and algorithms for 1D heat conduction |
| [Radiation Exchange](docs/radiation-exchange.md) | View factors, H-matrix compression, radiosity solver |
| [GPU Acceleration](docs/gpu-implementation-plan.md) | OpenCL/CUDA/HIP backend usage and architecture |
| [API Reference](docs/api-reference.md) | C++ class and function documentation |
| [Configuration Guide](docs/configuration.md) | Complete YAML configuration reference |

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

The thermal model implemented in PTM3D is based on methods described in:

- Hayne, P. O., Bandfield, J. L., Siegler, M. A., Vasavada, A. R., Ghent, R. R., Williams, J.-P., Greenhagen, B. T., Aharonson, O., Elder, C. M., Lucey, P. G., & Paige, D. A. (2017). Global regolith thermophysical properties of the Moon from the Diviner Lunar Radiometer Experiment. *Journal of Geophysical Research: Planets*, 122(12), 2371–2400. https://doi.org/10.1002/2017JE005387

- Hayne, P. O., & Aharonson, O. (2015). Thermal stability of ice on Ceres with rough topography. *Journal of Geophysical Research: Planets*, 120(9), 1567–1584. https://doi.org/10.1002/2015JE004887

- Vasavada, A. R., Bandfield, J. L., Greenhagen, B. T., Hayne, P. O., Siegler, M. A., Williams, J.-P., & Paige, D. A. (2012). Lunar equatorial surface temperatures and regolith properties from the Diviner Lunar Radiometer Experiment. *Journal of Geophysical Research: Planets*, 117(E12). https://doi.org/10.1029/2011JE003987

- Paige, D. A., Siegler, M. A., Zhang, J. A., Hayne, P. O., Foote, E. J., Bennett, K. A., Vasavada, A. R., Greenhagen, B. T., Schofield, J. T., McCleese, D. J., Foote, M. C., DeJong, E., Bills, B. G., Hartford, W., Murray, B. C., Allen, C. C., Snook, K., Soderblom, L. A., Calcutt, S., ... Lucey, P. G. (2010). Diviner lunar radiometer observations of cold traps in the Moon's south polar region. *Science*, 330(6003), 479–482. https://doi.org/10.1126/science.1187726

- Hayne, P. O., Greenhagen, B. T., Foote, M. C., Siegler, M. A., Vasavada, A. R., & Paige, D. A. (2010). Diviner lunar radiometer observations of the LCROSS impact. *Science*, 330(6003), 477–479. https://doi.org/10.1126/science.1197135

The H-matrix view factor compression is based on:

- Potter, S. F., Bertone, S., Schörghofer, N., & Mazarico, E. (2023). Fast hierarchical low-rank view factor matrices for thermal irradiance on planetary surfaces. *Journal of Computational Physics: X*, 17, 100130. https://doi.org/10.1016/j.jcpx.2023.100130

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [Sam Potter](https://github.com/sampotter/fluxpy) for H-matrix view factor algorithms and [fluxpy](https://github.com/sampotter/fluxpy)
- NASA NAIF for SPICE toolkit and ephemeris data
- Intel for Embree ray tracing library
- Lunar Reconnaissance Orbiter Diviner team for validation data
- NASA Lunar Vulkan Imaging and Spectroscopy Explorer (Lunar-VISE)
- NASA Europa Clipper mission
- NASA Precursor Science Investigations for Europa (PSIE)

---

**Questions?** Open an issue on GitHub or contact Paul Hayne (paul.hayne@colorado.edu).
