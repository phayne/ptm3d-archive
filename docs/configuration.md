# Configuration Guide

This document describes all configuration options for PTM3D simulations.

## YAML Configuration Files

PTM3D supports YAML configuration files for easier setup of complex simulations.

### Basic Structure

```yaml
simulation:
  body: "Europa"
  start_time: "2025-01-01T00:00:00"
  duration_days: 3.55
  timesteps: 144

mesh:
  resolution: 5
  use_topographic: false
  import_file: ""

thermal:
  run_thermal_mesh: true
  threads: 0
  equilibration_orbits: 20

illumination:
  include_primary_reflection: true
  use_batch_processing: true

radiation:
  enable: false
  use_hmatrix: false
  coupling_iterations: 10
  temperature_tolerance: 0.5

output:
  directory: "output/simulation"
  format: "binary"
  vtk_mode: "both"
```

## Configuration Sections

### simulation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `body` | string | "Moon" | Planetary body name |
| `start_time` | string | "2025-01-01T00:00:00" | Start time (ISO 8601) |
| `duration_days` | float | 1.0 | Simulation duration |
| `timesteps` | int | 24 | Number of output timesteps |

### mesh

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `resolution` | int | 5 | Global mesh resolution |
| `use_topographic` | bool | false | Use topographic mesh |
| `import_file` | string | "" | Path to mesh file (PLY/OBJ/VTK) |
| `scale_factor` | float | 1.0 | Mesh scale factor |

### thermal

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `run_thermal_mesh` | bool | false | Enable thermal model |
| `threads` | int | 0 | Thread count (0 = auto) |
| `equilibration_orbits` | int | 10 | Equilibration cycles |
| `use_gpu` | bool | false | Use GPU solver |
| `m` | int | 10 | Layers per skin depth |
| `n` | int | 15 | Layer growth factor |
| `b` | int | 8 | Skin depths to bottom |

### illumination

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_primary_reflection` | bool | true | Include reflected light |
| `use_batch_processing` | bool | true | Use SIMD batching |
| `albedo_map` | string | "" | Path to albedo map |

### radiation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable` | bool | false | Enable radiation exchange |
| `use_hmatrix` | bool | false | Use H-matrix compression |
| `coupling_iterations` | int | 10 | Max coupling iterations |
| `temperature_tolerance` | float | 0.5 | Convergence threshold (K) |
| `flux_tolerance` | float | 0.1 | Flux threshold (W/m²) |
| `view_factor_file` | string | "" | View factor cache file |

### output

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `directory` | string | "output" | Output directory |
| `format` | string | "binary" | Output format |
| `vtk_mode` | string | "both" | VTK output mode |
| `include_primary_ir` | bool | false | Include IR in VTK |

## Generating Configuration Templates

```bash
./bin/ptm_thermal_sim --generate-template my_config.yaml
```

## Overriding Configuration

Command-line options override YAML settings:

```bash
./bin/ptm_thermal_sim --config config.yaml --duration 7.0 --timesteps 168
```

## Supported Bodies

| Body | Day Length | Year Length | Distance (AU) |
|------|------------|-------------|---------------|
| Moon | 29.53 days | 365.25 days | 1.0 |
| Mercury | 176 days | 87.97 days | 0.387 |
| Europa | 3.55 days | 11.86 years | 5.2 |
| Enceladus | 1.37 days | 29.46 years | 9.54 |
| Mars | 1.03 days | 687 days | 1.52 |
| Ceres | 9.07 hours | 4.6 years | 2.77 |
