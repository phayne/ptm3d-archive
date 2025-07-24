# PTM: Planetary Thermophysical Model
A high-performance coupled illumination-thermal model for simulating surface temperatures on planetary bodies.

## Overview
PTM3D (Planetary Thermophysical Model - 3D) is a comprehensive computational framework that combines ray-traced illumination with advanced thermophysical modeling to simulate surface temperatures on planetary bodies. The model accounts for direct solar radiation, reflected sunlight from primary bodies (like Jupiter), infrared radiation, and internal heat flux to produce accurate thermal simulations.

## Key Features
* Coupled Illumination-Thermal Physics: Combines ray-traced illumination with advanced thermophysical modeling
* High-Performance Computing: Optimized with SIMD instructions, multi-threading via OpenMP, and efficient algorithms
* Accurate Planetary Positions: Uses SPICE kernels for precise planetary ephemerides
* Spatial Albedo Maps: Support for planetary albedo maps in FITS format
* Variable Resolution: Configurable mesh resolution for balancing accuracy and computational resources
* Visualization Tools: Python-based visualization and analysis tools
* Multiple Output Formats: VTK, binary, and CSV outputs for detailed analysis

## Installation
### Prerequisites
* C++ Compiler: GCC 8+ or Clang 10+ with C++17 support
* CMake: 3.12 or higher
* SPICE Toolkit: NASA NAIF SPICE toolkit
* Embree: Intel Embree ray tracing kernels (version 4)
* CFITSIO: Library for reading FITS format files
* OpenMP: For parallel processing
* Python 3.7+: For visualization tools (with NumPy, PyVista, Matplotlib)

### Building from Source
1. Clone the repository:
```git clone https://github.com/username/ptm.git
cd ptm```

2. Create a build directory:
```mkdir build
cd build```

3. Configure with CMake:
```cmake ..```

4. Build the project:
```make -j$(nproc)```

5. Install (optional):
```make install```

## Project Structure
The PTM codebase is organized into several key modules:

### Core Modules
* Illumination Model: Ray-traced solar illumination with shadowing effects
* Thermal Model: Thermophysical calculations for surface and subsurface temperatures
* Mesh Utilities: Tools for creating and managing planetary surface meshes
* Spatial Maps: Support for albedo and thermal property maps

### Tools
* Albedo Map Generator: Precomputes albedo maps for different mesh resolutions
* Map Processor: Utility for working with planetary maps in various formats
* VTK Converter: Converts simulation results to ParaView-compatible formats

### Python Components
* Thermal Visualization: Tools for visualizing temperature distributions
* Illumination Analysis: Analysis tools for solar illumination patterns
* ParaView Converter: Converts binary outputs to ParaView format

## Usage
### Basic Thermal Simulation

```./bin/ptm_thermal_sim --body Europa --duration 3.55 --equilibration-orbits 1 \
    --albedo-maps-directory /path/to/albedo_maps --thermal-mesh \
    --thermal-threads 48 --output /path/to/output --resolution 6```

### Command-Line Options
* --body: Target planetary body (Europa, Moon, Mercury, etc.)
* --duration: Simulation duration in days
* --resolution: Mesh resolution (higher values = more detailed mesh)
* --thermal-mesh: Enable thermal modeling for the entire mesh
* --thermal-threads: Number of threads for thermal computation
* --equilibration-orbits: Number of orbits for thermal equilibration
* --albedo-maps-directory: Directory containing albedo maps
* --output: Output directory for simulation results

## Modules
### Illumination Module
The illumination module calculates solar flux across the planetary surface, accounting for:

* Direct solar radiation
* Primary body shadows
* Reflected solar radiation from the primary body
* Infrared radiation from the primary body
* Optimization includes SIMD acceleration, efficient ray-batching, and multi-threading.

### Thermal Module
The thermal module simulates heat transfer and temperature evolution, including:

* Surface and subsurface temperatures
* Variable material properties with depth
* Internal heat flux
* Detailed boundary conditions
* Heat capacity variation with temperature

### Mesh Utilities
Handles mesh generation and processing:

* Icosphere-based global meshes at variable resolutions
* Custom mesh import from standard formats
* Mesh properties calculation (normals, centers, lat/lon mapping)
* Efficient data structures for ray-tracing

### Maps Module
Handles spatial data like albedo maps:

* FITS format support
* Coordinate transformations
* Bilinear interpolation
* Spatial queries

## Example Workflow
1. Generate or import an albedo map:

```./bin/albedo_map_generator input_albedo.fits output_dir 3 8```

2. Run a thermal simulation:

```./bin/ptm_thermal_sim --body Europa --duration 3.55 --equilibration-orbits 1 \
    --albedo-maps-directory /path/to/albedo_maps --thermal-mesh \
    --output /path/to/output --resolution 6```

3. Convert results to ParaView format:

```python src/python/ptm_paraview_converter.py --input-dir /path/to/output \
    --output-dir /path/to/paraview_output```

4. Visualize specific locations:

```python src/python/illumination_to_thermal.py --vtk /path/to/output/thermal.vtk \
    --locations locations.txt --output thermal_locations```

## License
This project is licensed under the MIT License.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (```git checkout -b feature/amazing-feature```)
3. Commit your changes (```git commit -m 'Add some amazing feature'```)
4. Push to the branch (```git push origin feature/amazing-feature```)
5. Open a Pull Request

## Acknowledgments
* NASA NAIF team for the SPICE toolkit
* Intel for the Embree ray tracing kernels
* The Lunar Vulkan Imaging and Spectroscopy Explorer

## Contact
For questions or feedback, please open an issue on GitHub or contact the maintainers.