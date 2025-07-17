# Planetary Thermophysical Model (PTM)

A computational model for simulating surface temperatures on planetary bodies including Europa, the Moon, Mercury, and others.

## Features

- Illumination modeling with ray-tracing for shadowing effects
- Thermophysical modeling with depth-dependent properties
- Support for multiple planetary bodies with appropriate physical parameters:
  - Europa (albedo: 0.5, emissivity: 0.9)
  - Moon (albedo: 0.12, emissivity: 0.95)
  - Mercury (albedo: 0.07, emissivity: 0.93)
- SPICE integration for accurate ephemeris data
- Variable thermal properties (conductivity, density) with depth

## Building and Installation

### Prerequisites
- C++17 compatible compiler
- CMake 3.12+
- SPICE toolkit
- FITS library for map handling

### Build Instructions
```bash
mkdir build
cd build
cmake ..
make

## Usage

./bin/ptm_thermal_sim --body Moon --thermal-mesh --thermal-threads 48 --resolution 5

## License

MIT

