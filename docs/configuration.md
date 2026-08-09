# Configuration Guide

This document lists the YAML configuration keys PTM3D reads, with the defaults the
parser applies when a key is absent (`src/cpp/tools/config_parser.cpp`).

> Runtime behavior is also influenced by `PTM_*` **environment variables** — see
> [ENVIRONMENT.md](ENVIRONMENT.md) for the complete table, including the
> `engine_env:` block that lets a configuration declare its own environment.

## Structure

A configuration file has up to eight top-level sections: `simulation`, `mesh`,
`thermal`, `illumination`, `radiation`, `output`, `generic_body` and
`thermophysical`, plus the optional `engine_env` block. Every section is
optional; absent keys take the defaults below.

```yaml
simulation:
  body: "Moon"
  start_time: "2025-01-01T00:00:00"
  duration_days: 29.53
  timesteps: 72

mesh:
  resolution: 5
  import_file: ""

thermal:
  run_thermal_mesh: true
  threads: 0
  equilibration_orbits: 4

illumination:
  include_primary_reflection: true
  use_batch_processing: true

radiation:
  enabled: true
  band_separated: true
  max_coupling_iterations: 10

output:
  directory: "output/simulation"
  format: "binary"
  vtk_mode: "both"
```

Runnable examples are in [`../examples/`](../examples/); start with
`quickstart_moon_diurnal.yaml`.

## simulation

| Key | Type | Default | Description |
|---|---|---|---|
| `body` | string | `"Europa"` | Planetary body name |
| `start_time` | string | `"2025-01-01T00:00:00"` | Start time (ISO 8601) |
| `duration_days` | float | 3.55 | Simulated duration of the output window |
| `timesteps` | int | 72 | Number of output timesteps |

## mesh

| Key | Type | Default | Description |
|---|---|---|---|
| `resolution` | int | 5 | Icosphere subdivision level when no mesh is imported |
| `import_file` | string | `""` | Mesh file to load (PLY/OBJ/VTK) |
| `occluder_file` | string | `""` | Shadow-only occluder mesh, added to the ray-tracing scene but never solved or written — see [extended-occluder-architecture.md](extended-occluder-architecture.md) |
| `scale_factor` | float | 1.0 | Vertex scale factor (see [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) before using it) |
| `export_basic` | bool | false | Write the generated mesh before solving |
| `use_topographic` | bool | false | Build the mesh from a topographic map |
| `topo_directory` | string | `""` | Directory holding the topography maps |
| `latitude_override` | float | -999.0 | Site latitude for topographic meshing (sentinel = unset) |
| `longitude_override` | float | -999.0 | Site longitude for topographic meshing (sentinel = unset) |

## thermal

| Key | Type | Default | Description |
|---|---|---|---|
| `run_thermal_mesh` | bool | false | Run the thermal solver (illumination only when false) |
| `threads` | int | 0 | OpenMP thread count (0 = auto) |
| `thermal_m` | int | 10 | Layers within the upper skin depth |
| `thermal_n` | int | 15 | Layer growth factor (legacy non-adaptive grid only) |
| `thermal_b` | int | 15 | Skin depths to the bottom layer |
| `equilibration_orbits` | float | 1.0 | Diurnal cycles of spin-up before the output window |
| `equilibration_steps_per_day` | float | -1.0 | Explicit equilibration cadence in steps per Earth-day (-1 = unset) |
| `equilibration_samples_per_diurnal_cycle` | float | -1.0 | Target illumination samples per diurnal cycle; consulted only when the explicit cadence is unset. 24 gives about 2 K accuracy on the Moon, 48 about 1 K |
| `equilibration_cycles` | int | -1 | Passes over the equilibration forcing array, GPU persistent solver only (-1 = the solver's value of 2) |
| `eq_extrapolation` | map | off | Pass-stride extrapolation of the equilibration state; see below |
| `compute_statistics` | bool | true | Compute and write per-face statistics |
| `use_gpu` | bool | false | OpenCL thermal solver (a CPU-only build refuses at startup) |
| `gpu_devices` | int | 1 | Devices in the thermal pool (0 = all enumerated) |
| `gpu_device_ids` | list[int] | [] | Explicit device indices; overrides `gpu_devices` |
| `gpu_batch_size` | int | 0 | Faces per GPU batch (0 = automatic cap) |
| `gpu_inter_batch_pause_ms` | int | 0 | Pause between consecutive batches on a device |
| `auto_calibrate_gpu` | bool | true | Probe the host once for a dispatch ceiling when no machine profile exists |
| `thermal_inertia_map` | string | `""` | Per-face `float32` thermal-inertia map |
| `h_parameter_map` | string | `""` | Per-face `float32` H-parameter map (mutually exclusive with the above) |
| `use_default_property_map` | bool | false | Auto-discover the bundled property map for the body |
| `property_maps_directory` | string | `"data/property_maps"` | Where to look for property maps |
| `geothermal_heat_flow` | float | -1.0 | Basal heat flow in W m<sup>-2</sup> (-1 = body default) |
| `heat_flow_mask_file` | string | `""` | Face-index file for locally elevated basal flux |
| `heat_flow_mask_value` | float | 1.0 | Basal flux on the masked faces (W m<sup>-2</sup>) |
| `integration_days` | float | -1.0 | Statistics-integration window (-1 = the output window) |
| `integration_steps` | int | -1 | Same window, expressed in steps |
| `integration_offset_days` | float | 0.0 | Offset of the integration window |
| `integration_offset_steps` | int | 0 | Same offset, expressed in steps |
| `initial_state_file` | string | `""` | Load a per-face `T(z)` checkpoint and skip equilibration |
| `final_state_file` | string | `""` | Write a per-face `T(z)` checkpoint at the end of the run |
| `checkpoint_every_output_steps` | int | 0 | Mid-run checkpoint cadence (0 = chunk boundaries only) |
| `compute_ice_stability` | bool | false | Compute the subsurface ice-stability depth |
| `enable_shadow_flux` | bool | false | Sub-grid crater shadowing correction |
| `crater_dd_ratio` | float | 8.0 | Diameter-to-depth ratio for that correction |
| `debug` | bool | false | Verbose per-layer output for one equatorial face |

The checkpoint files are used by the chunked-run workflow
(`scripts/run_chunked_simulation.py` and `ptm_merge_chunks`).


### Equilibration extrapolation (`thermal.eq_extrapolation`)

The GPU equilibration phase replays one forcing block several times
(`equilibration_cycles`). Because the same block is replayed unchanged, the
temperature field after each pass is a fixed-point iteration, and its limit can
be estimated from a few passes instead of reached by brute force. Enabling this
option applies reduced-rank extrapolation to that sequence and then runs the
final pass from the extrapolated state, so the last pass serves as polish,
verification and delivered state at once.

Default is `off`, and with it off the solver takes the pre-existing code path
and produces byte-identical output.

```yaml
thermal:
  equilibration_cycles: 5
  eq_extrapolation:
    method: rre        # off | rre          (default off)
    order_p: 4         # 3 or 4; needs equilibration_cycles >= order_p + 1
    tol_K: 0.5         # certificate reporting threshold, K
    hold_K: 1.0e-3     # per-pass increment below which a layer is held, K
```

`order_p` below 3 is refused rather than adjusted: a second-order fit is less
accurate than not extrapolating at all. If `equilibration_cycles` is smaller
than `order_p + 1` the run warns and uses plain cycling.

Every run reports a one-line summary of how many faces were extrapolated, how
many the certificate declined, the 95th-percentile error estimate, the fraction
of layers held, and the number of near-degenerate windows.

**Accuracy and where it applies.** On a 509,337-face lunar polar mesh at
`order_p: 4` with five passes, the extrapolated state was 55x closer (sunlit
faces) and 64x closer (seasonal-terminator faces) to a 200-lunation reference
than four passes of plain cycling, for 1.18x the wall-clock of four passes. The
benefit accrues to faces that carry energy; about 70 percent of that mesh is
already converged at depth after four passes and gains nothing.

**A known limitation.** The fit assigns a weight to the first snapshot in its
window, which is the initial guess. For a depth layer that converged early while
shallower layers were still evolving, that weight can pull the layer back toward
the initial guess. The displacement is bounded by

    displacement  <=  |gamma_0| * (initial-guess error at that depth)

with `gamma_0` the weight on the first snapshot, measured at about 0.09 - so
roughly ten percent of the initial-guess error, not an amplification of it. The
`hold_K` test suppresses this by holding any layer whose last-pass increment is
below the threshold, and the run reports how many windows remain susceptible.
On the mesh above, 73 of 509,337 faces retained a displacement, the largest
0.47 K at 1 m depth, all on faces receiving no flux during equilibration.
Raising `hold_K` holds more layers and trades extrapolation speed-up for
robustness; the layers it holds simply converge at the ordinary rate.
## illumination

| Key | Type | Default | Description |
|---|---|---|---|
| `include_primary_reflection` | bool | true | Include reflected sunlight and thermal IR from the primary body |
| `use_batch_processing` | bool | true | Trace illumination rays in vectorized batches (faster, same results) |
| `use_albedo_map` | bool | true | Load a spatially varying albedo map when one is available |
| `albedo_map` | string | `""` | Explicit albedo FITS map |
| `albedo_maps_directory` | string | `"data/albedo_maps"` | Where to look for albedo maps |
| `positions_file` | string | `""` | Precomputed body/Sun positions instead of SPICE lookups |
| `use_gpu` | bool | false | GPU BVH shadow-ray tracer |
| `gpu_devices` | int | 1 | Devices for the ray tracer (0 = all enumerated) |
| `gpu_device_ids` | list[int] | [] | Explicit device indices |
| `gpu_inter_batch_pause_ms` | int | 0 | Pause between consecutive batches on a device |
| `bvh_cache_file` | string | `""` | BVH cache path (empty = auto under the output directory; `off` disables) |

## radiation

Radiation exchange is configured from YAML as well as from the command line; the
command line wins. The full key list, with the physics behind each, is in
[radiation-exchange.md §9](radiation-exchange.md). The most commonly used keys:

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | false | Enable inter-face radiation exchange |
| `band_separated` | bool | **true** | Band-separated visible + thermal-IR radiosity; false = single gray band |
| `exchange_selection` | string | `all` | Which faces keep the computed indirect flux (`all` or `shadow`) |
| `view_factor_file` | string | `""` | Save/load path for the view-factor matrix; a relative path resolves against the configuration file's directory |
| `use_hmatrix` | bool | false | Hierarchical-matrix compression for large meshes |
| `use_monte_carlo_vf` | bool | false | Monte Carlo ray-gather view-factor builder |
| `coupling_mode` | string | `bounded` | `bounded`, `fixed` or `legacy` |
| `max_coupling_iterations` | int | 10 | Picard iteration cap |
| `coupling_temp_threshold` | float | 0.5 | Temperature convergence threshold (K) |
| `use_gpu` | bool | false | OpenCL radiosity solver |
| `verbose` | bool | false | Detailed radiation progress |

## output

| Key | Type | Default | Description |
|---|---|---|---|
| `directory` | string | `"output"` | Output directory |
| `format` | string | `"binary"` | `binary` or `vtk` |
| `vtk_mode` | string | `"both"` | `thermal`, `complete` or `both` |
| `include_primary_ir` | bool | false | Include the primary-body IR field in VTK output |
| `thermal_vtk_file` | string | `""` | Explicit thermal VTK path |
| `complete_vtk_file` | string | `""` | Explicit complete VTK path |
| `locations_file` | string | `""` | CSV of point locations to export |
| `series_output` | string | `""` | Time-series output path |
| `location_data` | string | `""` | Per-location output path |
| `depths_file` | string | `""` | Depth list for location exports |
| `skip_time_series` | bool | false | Skip per-step binaries; statistics are still computed |
| `write_raw_stats` | bool | false | Write merge-friendly raw statistics for the chunked workflow |
| `skip_mesh` | bool | false | Do not copy `mesh.ply` into the output directory |

## thermophysical

On a built-in body (Moon, Mercury, Europa) the parser consumes only the keys
below. Plain-named properties such as `ks`, `kd`, `H`, `rhos`, `cp0` or `albedo`
are read only under `generic_body.surface:`; on a named body the run prints one
`[config] WARNING:` line naming every such key it found and does not apply them.
See [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md).

| Key | Type | Default | Description |
|---|---|---|---|
| `use_default_albedo` | bool | true | false marks the albedo as explicitly configured |
| `constant_thermal_inertia` | float | -1.0 | Above 0 selects constant-thermal-inertia mode, which forces the radiative conductivity term to zero |
| `albedo_override` | float | -1.0 | Bond albedo override (sentinel = body default) |
| `emissivity_override` | float | -1.0 | Emissivity override |
| `ks_override` | float | -1.0 | Surface conductivity override |
| `kd_override` | float | -1.0 | Deep conductivity override |
| `incidence_angle_albedo` | bool | false | Enable the Hapke directional-hemispheric albedo |
| `albedo_model` | string | `""` | `hapke` is equivalent to the key above |
| `albedo_hapke_w` | float | -1.0 | Single-scattering albedo (below 0 = derive from the Bond albedo) |
| `conductivity_model` | string | `"hayne2017"` | `hayne2017` or `martinez2021` — see [martinez-conductivity-model.md](martinez-conductivity-model.md). An unrecognized value stops the run |

## generic_body

Bodies without built-in properties are described here, with an ephemeris from
either Keplerian elements or a SPICE metakernel, plus a `surface:` block that
carries the full thermophysical property set. See the asteroid examples in
[`../examples/`](../examples/) and
[docs/tutorials/](tutorials/).

## Generating a template

```bash
./build/bin/ptm_thermal_sim --generate-template my_config.yaml
```

## Overriding from the command line

Command-line options override YAML settings:

```bash
./build/bin/ptm_thermal_sim --config config.yaml --duration 7.0 --timesteps 168
```

The full flag list is in [USER_GUIDE.md](USER_GUIDE.md) and from
`ptm_thermal_sim --help`.

## Built-in bodies

| Body | Day length | Year length | Distance (AU) |
|---|---|---|---|
| Moon | 29.53 days | 365.25 days | 1.0 |
| Mercury | 176 days | 87.97 days | 0.387 |
| Europa | 3.55 days | 11.86 years | 5.2 |

These three have pre-defined thermophysical property sets. Any other body name
falls back to placeholder defaults and expects a `generic_body:` block to supply
its properties; the run says so on stderr.
