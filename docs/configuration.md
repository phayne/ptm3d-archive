# Configuration Guide

This document lists the YAML configuration keys PTM3D reads, with the defaults the
parser applies when a key is absent (`src/cpp/tools/config_parser.cpp`).

> Changing a key or a default in the parser? Regenerate the Python layer's copy of
> this table as well — see [CONTRIBUTING.md](../CONTRIBUTING.md#changing-an-engine-configuration-key).

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
  max_coupling_iterations: 16

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
| `start_time_offset_seconds` | float | 0.0 | Ephemeris-time offset added after `start_time` is resolved (see [The output epoch grid](#the-output-epoch-grid)) |
| `duration_days` | float | 3.55 | Simulated duration of the output window |
| `timesteps` | int | 72 | Number of output timesteps |

### The output epoch grid

The output grid is **half-open**. A window of `timesteps` samples over
`duration_days` has cadence

```
dt = duration_days * 86400 / timesteps      [ephemeris seconds]
```

and samples `start_time + i*dt` for `i` in `[0, timesteps)`. The last sample
therefore sits one cadence *before* the end of the window, and the next window
of the same simulation begins exactly at `start_time + timesteps*dt`. That is
what lets consecutive windows tile: no epoch is sampled twice and none is
skipped.

An equilibration prefix does not disturb this. `equilibration_orbits` adds its
own samples *before* `start_time`, so output frame 0 is at `start_time` exactly,
whatever the equilibration length or cadence.

`start_time_offset_seconds` shifts the whole grid by a number of **ephemeris**
seconds. Use it, rather than a shifted UTC `start_time` string, whenever a
window boundary has to land on an exact multiple of the cadence: TDB − UTC
carries a periodic relativistic term (~1.7 ms amplitude, annual period) and
steps by a whole second at a leap second, so adding to the UTC string is not the
same shift. `scripts/run_chunked_simulation.py` uses it for every chunk.

> Changed 2026-08-12. Before that date the default (uniform-cadence) path
> computed `dt = (equilibration_days + duration_days) * 86400 / (total_steps −
> 1)`: the output cadence was set by dividing a span that included the
> equilibration prefix, by one fewer interval than there are samples. Output
> epochs on every run shifted as a result — on an 80 N lunar crater with 236
> steps over 59.06 days and one equilibration lunation, frame 0 was 1 h 59 m
> 53.7 s after the nominal start and the cadence was 0.28 % long. Products
> generated before that date carry those labels. See
> [chunked-run-equivalence.md](chunked-run-equivalence.md).

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
| `thermal_n` | int | 15 | Inert. The legacy non-adaptive grid it feeds cannot be selected from any config key, flag or environment variable, so this value and the `--thermal-n` flag have no effect |
| `thermal_b` | int | 15 | Skin depths to the bottom layer |
| `equilibration_orbits` | float | 1.0 | **Integrated** diurnal cycles of real, ephemeris-driven spin-up before the output window. The same number on the CPU scalar, CPU SIMD and GPU persistent paths; `equilibration_cycles` changes how it is reached, never the total. The CPU paths additionally run a fixed analytic pre-conditioner that this key does not count — see [Equilibration length](#equilibration-length-what-the-number-means). Choose it with [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) section 1 |
| `equilibration_steps_per_day` | float | -1.0 | Explicit equilibration cadence in steps per Earth-day (-1 = unset) |
| `equilibration_samples_per_diurnal_cycle` | float | -1.0 | Target illumination samples per diurnal cycle; consulted only when the explicit cadence is unset. 24 gives about 2 K accuracy on the Moon, 48 about 1 K |
| `equilibration_cycles` | int | -1 | Replays of the equilibration forcing block used to reach `equilibration_orbits`, GPU persistent solver only (-1 = 1 replay). The block is `equilibration_orbits / equilibration_cycles` cycles long, so the integrated total does not change and only the flux RAM and illumination cost fall |
| `equilibration_target_K` | float | 0.1 | Accuracy the spin-up estimate targets, and the bar the retrospective certificate must clear — see [Did the column settle?](#did-the-column-settle-the-retrospective-certificate) |
| `equilibration_stop_on_certificate` | bool | false | End the analytic pre-conditioner when the certificate reaches `equilibration_target_K` instead of at a fixed cycle count. CPU SIMD batch path only; the other solver paths refuse to start with it set |
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
| `geothermal_heat_flow` | float | -1.0 | Basal heat flow in W m<sup>-2</sup> (-1 = body default). **Discarded on built-in bodies** on both backends, while the log announces it — use `heat_flow_mask_file`, or `generic_body.surface.Qb` on a generic body. See [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) section 4 |
| `heat_flow_mask_file` | string | `""` | Face-index file for locally elevated basal flux |
| `heat_flow_mask_value` | float | 1.0 | Basal flux on the masked faces (W m<sup>-2</sup>) |
| `integration_days` | float | -1.0 | Statistics-integration window (-1 = the output window) |
| `integration_steps` | int | -1 | Same window, expressed in steps |
| `integration_offset_days` | float | 0.0 | Offset of the integration window |
| `integration_offset_steps` | int | 0 | Same offset, expressed in steps |
| `initial_state_file` | string | `""` | Load a per-face `T(z)` checkpoint and skip equilibration (see [Thermal-state checkpoints](#thermal-state-checkpoints)) |
| `final_state_file` | string | `""` | Write a per-face `T(z)` checkpoint at the end of the run |
| `checkpoint_lookahead_steps` | int | 0 | Extra output steps integrated past the export window, so `final_state_file` is the state at the NEXT window's first epoch (see [Thermal-state checkpoints](#thermal-state-checkpoints)) |
| `checkpoint_every_output_steps` | int | 0 | Mid-run checkpoint cadence (0 = chunk boundaries only) |
| `eq_indirect_source` | string | `auto` | Where the equilibration phase gets its indirect flux: `auto` estimates it from the coupling loop's first sweep, `converged` from its converged sweep (better and ~1.9× the wall clock), `off` is the pre-2026-08 behavior ([radiation-exchange.md §7.7](radiation-exchange.md)) |
| `eq_indirect_passes` | int | 1 | How many times that estimate is refreshed and the equilibration repeated. Must be ≥ 1. A second `auto` pass is that estimator's knee but the poorest value on offer — spend on `eq_indirect_source: converged` first ([radiation-exchange.md §7.7](radiation-exchange.md)) |
| `eq_indirect_flux_file` | string | `""` | Explicit per-face `'QIND'` mean-indirect file; overrides `eq_indirect_source` |
| `indirect_mean_out_file` | string | `""` | Write the converged per-face output-window mean indirect flux as a `'QIND'` file |
| `compute_ice_stability` | bool | false | Compute the subsurface ice-stability depth |
| `enable_shadow_flux` | bool | false | Sub-grid crater shadowing correction |
| `crater_dd_ratio` | float | 8.0 | Diameter-to-depth ratio for that correction |
| `debug` | bool | false | Verbose per-layer output for one equatorial face |

The checkpoint files are used by the chunked-run workflow
(`scripts/run_chunked_simulation.py` and `ptm_merge_chunks`).


### Thermal-state checkpoints

`thermal.initial_state_file` and `thermal.final_state_file` hand a whole-mesh
per-face `T(z)` column between runs. Every whole-mesh solver path reads and
writes them — the GPU persistent solver, the CPU SIMD batch solver, the CPU
scalar solver, and the streamed CPU path — in the same format
(`uint32` magic `0x50544D54` `'PTMT'`, `uint32 n_faces`, `uint32 n_layers`, then
`n_layers × n_faces` `float32` in layer-major order). A checkpoint written by
either backend restores on the other.

Three things to know before relying on one.

- **It refuses; it does not degrade.** A missing file, a wrong magic number, a
  truncated body, a face- or layer-count mismatch, or a destination that cannot
  be opened for writing aborts the run with a message naming the reason. It
  never falls back to a cold start, because a silently cold-started chunk
  reports spin-up as physics.
- **The file is `float32`, on every path.** The CPU solvers carry `T` in
  `float64`, so writing a checkpoint rounds each value once, at the write.
  Restoring is exact. A CPU → file → CPU round trip therefore reproduces the
  file bit-for-bit and the in-memory column to `float32` precision.
- **The file does not record the depth grid.** It carries the shape and nothing
  else, so a checkpoint written on one grid will load onto a different grid that
  happens to have the same layer count. The engine prints the live grid
  (`n_layers`, `z[0]`, `dz[0]`, `z_max`) at both the write and the restore, and
  says in the restore message which quantities it could not verify — compare the
  two lines when resuming across a configuration change.

Setting `initial_state_file` forces `equilibration_orbits` to 0 and disables the
analytic clear-sky pre-conditioner: a restored column is a continuation, and
spinning it up again would discard it. Both are announced in the log.

**Where the checkpoint sits in time.** The solver integrates
`total_time_steps − 1` intervals, so by default `final_state_file` holds the
state at the *last exported frame*. On a half-open grid the next window begins
one cadence later than that, so a handoff needs one extra grid point.
`thermal.checkpoint_lookahead_steps: 1` supplies it: the run integrates one
output step past the export window and checkpoints there, and that extra step is
neither written as a frame nor reduced into `statistics/`. It is also kept out
of the equilibration seed's forcing mean, so it does not change the run's
initial condition. `scripts/run_chunked_simulation.py` sets it for every chunk;
outside the chunked workflow you do not need it.

One case is disclosed rather than refused. With radiation coupling on a CPU
solver, the coupled re-run keeps no `T(z)` column, so the checkpoint holds the
**un-coupled** column on the faces the re-run touched. The run says so.


### Equilibration length: what the number means

`thermal.equilibration_orbits` is the **integrated** number of diurnal cycles of
real, ephemeris-driven forcing the solver runs before the output window opens.
That is true on all three solver paths and at any `equilibration_cycles`:

```
block_cycles = equilibration_orbits / equilibration_cycles
integrated   = equilibration_cycles * block_cycles = equilibration_orbits
```

`equilibration_cycles` is a memory and cost dial, not a length. Raising it
shortens the assembled forcing block and replays it more times, which cuts the
equilibration flux array and the illumination trace by that factor while
integrating the same number of cycles. That is the only affordable route past
the flux-array size limit: a 25-cycle equilibration block is about 6.1 GB on a
509,337-face lunar mesh. It is a trade, not a free lunch — *N* replays of a
short block are not the same forcing as one long block of real ephemeris, and
the block should be a whole number of diurnal cycles so each replay restarts at
the same point in the cycle it ended on. The engine warns when it is not.

Only the GPU persistent solver has a replay loop. On a CPU run the engine prints
a notice and delivers the same integrated length as one block of the full size,
which costs more memory and traces more illumination but is the same spin-up.

Every run prints all of this before it starts, so the configured number and the
integrated number can be compared directly:

```
Equilibration (configured): 20.00 day/night cycles  (thermal.equilibration_orbits)
Equilibration (integrated): 20.00 day/night cycles = 2 pass(es) x 10.00-cycle forcing block
```

**What `equilibration_orbits` does not count.** The CPU scalar and CPU SIMD
paths run a fixed analytic clear-sky pre-conditioner before the real forcing —
`NYEARSEQ * round(year/day)` diurnal cycles, which is 24 on the Moon, 2 on
Europa and 4 on Mercury. It uses a flat, unshadowed diurnal insolation with the
body-average albedo, it is not configurable from any key, and the GPU path has
no equivalent. The pre-flight banner prints its length on the CPU paths. So on
the GPU the integrated spin-up is exactly `equilibration_orbits`, and on the CPU
it is `equilibration_orbits` of real forcing on top of that fixed analytic seed.

**Before 2026-08-11** `equilibration_orbits` was the *block* length and the GPU
pass count defaulted to 2, so a GPU run integrated twice what its config said
while a CPU run integrated once. `equilibration_orbits: 25` bought 50 lunations
on the GPU. The shipped GPU configurations in `configs/` and `examples/` were
rewritten at the same time as this change to state the integrated number they
had always been running (10 became 20, and so on) with `equilibration_cycles: 2`
alongside, so their results are unchanged. A configuration written before that
date and not rewritten will now integrate half of what it used to on the GPU;
double its `equilibration_orbits` to restore the old behavior.

### Did the column settle? The retrospective certificate

`equilibration_orbits` says how long to spin up. It does not say whether that
was enough. The certificate answers the second question, and it answers it from
a conservation law rather than from how much the temperature is still moving.

The quantity is the cycle-mean interface heat flux at a cut depth
`z_c = min(5 delta_ref, z_max/2)`, measured against the geothermal flux the
column is driven by, and converted to kelvin:

```
r_c   = qbar_c - Qb                       [W/m^2]
E     = |r_c| / h,   h = 4 eps sigma T^3
```

Below the forced layer a settled column carries exactly `Qb` at every depth,
because nothing there stores or releases net heat over a cycle. `r_c` is
therefore a statement about the state's spatial gradient, not about its rate of
change, and that distinction is the point: when float32 rounding freezes a
column it stops changing without having converged, so "run until it stops
moving" and pass-to-pass bit identity both report success at exactly the moment
they have failed. On the banked 8-pass lunar state — bit-frozen across 100
further lunations — the certificate still reports 0.26 K against a 0.1 K
target.

Which `h` depends on the product:

| Observable | `h` |
|---|---|
| diurnal maximum | `4 eps sigma T_max^3` |
| diurnal minimum | `4 eps sigma T_min^3` |
| diurnal mean | the **harmonic** cycle mean of `4 eps sigma T0(t)^3` |
| deep products | `E_deep = |r_c| tau1 / C_below` (unbiased, **not** conservative — pointwise 0.91–1.09; reported, never used to pass) |

The mean uses the harmonic cycle mean and not the arithmetic one. The surface
displacement caused by a flux residual is largest exactly where the radiative
stiffness is smallest, so the two are anti-correlated and the arithmetic mean
understates the error — measured at 8.7x on a 25 %/75 % duty cycle between
350 K and 100 K. The minimum is in the table for the same reason: it is the
slackest instant of the cycle and therefore the most flux-sensitive surface
observable. Gating a lunar spin-up on the mean alone stopped it 13.6 cycles
early and left `temperature_min` 0.69 K from its full-length answer while
`temperature_max` moved 0.03 K; gating on all three left it at 0.06 K.

**After a run — `build/bin/ptm_eq_certify`.** The certificate runs against a
banked `PTMT` checkpoint and the statistics beside it, with no mesh, no view
factors and no GPU, so it can audit a finished run (or a running one's last
chunk) while another job owns the devices:

```
build/bin/ptm_eq_certify --config <run>/chunk_000/config.yaml \
    --state <run>/_thermal_state/thermal_state_after_chunk_000.bin \
    --stats <run>/chunk_000/statistics --period-ref annual
```

Exit status 0 = every scored face passed, 2 = at least one FAIL, 3 = at least
one INDETERMINATE, 1 = the tool refused to run. It will not call a snapshot of
a forced column a cycle mean, and it will not substitute the arithmetic mean of
`h` for the harmonic one; where the data cannot decide it reports
INDETERMINATE and says what would settle it. `Qb` is read out of the state's
own bottom boundary condition rather than from the config, because
`geothermal_heat_flow` was inert on every solver entry point for the whole
moon3d campaign.

**During a run — `thermal.equilibration_stop_on_certificate`.** Default false.
When true, the analytic clear-sky pre-conditioner ends when all three surface
observables reach `equilibration_target_K` instead of at its fixed cycle count,
and the achieved `E` is reported either way. It governs only that phase: the
real-ephemeris equilibration prefix cannot end early, because its steps are
epochs on the output grid and stopping it would move the output window rather
than save spin-up. The key is implemented on the CPU SIMD batch path; every
other entry point, and a run with no pre-conditioner enabled, refuses to start
with it set rather than quietly integrating a fixed count.

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
  equilibration_orbits: 50   # integrated diurnal cycles
  equilibration_cycles: 5    # ...as 5 replays of a 10-cycle block
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
benefit accrues to faces that carry energy.

About 70 percent of that mesh receives no flux during equilibration, and those
faces gain nothing from extrapolation — but not because they have converged.
On the GPU path their deep columns stop changing between passes while still
about 0.36 K short of the steady state, so extrapolating a sequence that has
already stopped moving has nothing to work with. Do not read "unchanged between
passes" as "converged"; see
[KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) sections 1 and 3.

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
| `max_coupling_iterations` | int | 16 | Picard iteration cap (raised from 10 on 2026-08-15; see radiation-exchange.md §7.5) |
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

### How `generic_body.spice_metakernel` is resolved

| Value | Resolved to |
|---|---|
| Absolute path (`/…/metakernel.tm`) | Used **exactly as given** — the kernel directory is not prepended |
| Relative `data/…` path (`data/horizons_cache/…/metakernel.tm`) | The same `PTM_DATA_ROOT` contract as `mesh.import_file` and every other data key: `$PTM_DATA_ROOT`, else the repository `data/` found from the config file's directory |
| Any other relative path (`bennu.tm`) | `<kernel dir>/<value>` first, then `<current working directory>/<value>` |

The kernel directory is itself resolved as: `$PTM_SPICE_KERNELS`, then
`$PTM_DATA_ROOT/kernels`, then the compiled-in `SPICE_KERNELS_DIR` (CMake cache
variable, default `<repo>/data/kernels`) — see
[ENVIRONMENT.md](ENVIRONMENT.md). There is no config key or command-line flag
for it in `ptm_thermal_sim` (`ptm_camera_sim` has `--kernels-path`); each rung
**replaces** the compiled-in directory rather than adding to it, so an override
directory must contain the metakernels as well as `lsk/`, `pck/` and `spk/`.

`$PTM_SPICE_KERNELS` is taken **exactly as given**, populated or not, because it
is an explicit instruction and demoting it would hide a typo. The two implicit
rungs are different: a directory with no `lsk/naif0012.tls` cannot serve any run
(not even a Keplerian one — `start_time` still has to be converted), so it is
passed over in favor of the next rung. That is what lets a fresh clone or
worktree, which ships the `*.tm` files but not `lsk/`, `pck/` or `spk/`, find a
kernel tree through `$PTM_DATA_ROOT` instead of dying several screens later in
`str2et_c` with a bare `SPICE(NOLEAPSECONDS)`.

Every run prints the kernel directory it resolved and its source (`SPICE kernel
directory: … ($PTM_DATA_ROOT/kernels)`) and the metakernel file it actually
loaded (`Loaded SPICE meta-kernel: …`). If none of the candidates exists the run
stops and lists every path it tried — both the metakernel candidates and the
whole kernel-directory ladder, in order, with the reason each rung was or was
not taken. It never substitutes a different metakernel for the one you
configured.

(Separately, and not covered by that guarantee: a `simulation.body` name other
than `Moon`, `Mercury` or `Europa` that is *not* backed by a `generic_body:`
block falls through to Europa's built-in property set and Europa's metakernel.
Use `generic_body:` for anything outside the three built-ins.)

Paths **inside** the metakernel (`PATH_VALUES`) are resolved by CSPICE itself,
relative to the working directory, and are independent of the rule above. The
shipped `data/kernels/*.tm` use `PATH_VALUES = ( 'data/kernels' )`, so a
built-in body expects to be run from the repository root; a metakernel with
absolute `PATH_VALUES` works from anywhere. A `PATH_VALUES` entry longer than
80 characters must be split with the KPL `+` continuation.

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
