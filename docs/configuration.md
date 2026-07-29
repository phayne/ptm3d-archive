# Configuration Guide

This document describes all configuration options for PTM3D simulations.

> Runtime behavior is also influenced by `PTM_*` **environment variables**
> (validated illumination fixes, radiosity band separation, memory/GPU knobs) —
> see [ENVIRONMENT.md](ENVIRONMENT.md) for the complete table, including the
> recommended-ON set used by production-validated runs.

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

# Note: radiation options are CLI-only, not in YAML

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

**Note:** Radiation options are passed via command-line arguments, not YAML configuration.

| CLI Flag | Default | Description |
|----------|---------|-------------|
| `--radiation-exchange` | off | Enable radiation exchange |
| `--rad-hmatrix` | off | Use H-matrix compression |
| `--rad-use-gpu` | off | Use GPU for radiosity solver |
| `--rad-verbose` | off | Print detailed progress |
| `--rad-view-factor-file <file>` | "" | Save/load view factors |
| `--rad-coupling-iter <n>` | 10 | Max coupling iterations |
| `--rad-coupling-temp-tol <K>` | 0.5 | Temperature convergence threshold |
| `--rad-coupling-flux-tol <W/m²>` | 0.1 | Flux change threshold |
| `--rad-coupling-face-frac <f>` | 0.01 | Fraction of faces still changing to stop |
| `--rad-aca-tolerance <tol>` | 0.001 | ACA relative tolerance |
| `--rad-aca-max-rank <n>` | 50 | Maximum rank for ACA |
| `--rad-use-svd` | off | Use SVD instead of ACA for compression |

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

Note: These are the currently supported bodies with pre-defined thermophysical properties in the C++ code.
