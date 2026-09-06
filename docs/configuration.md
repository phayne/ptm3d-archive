# Configuration Guide

This document lists the YAML configuration keys PTM3D reads, with the defaults the
parser applies when a key is absent (`src/cpp/tools/config_parser.cpp`).

> Changing a key or a default in the parser? Regenerate the Python layer's copy of
> this table as well — see [CONTRIBUTING.md](../CONTRIBUTING.md#changing-an-engine-configuration-key).

> Runtime behavior is also influenced by `PTM_*` **environment variables** — see
> [ENVIRONMENT.md](ENVIRONMENT.md) for the complete table, including the
> `engine_env:` block that lets a configuration declare its own environment.

## Structure

A configuration file has up to ten top-level sections: `simulation`, `mesh`,
`thermal`, `illumination`, `radiation`, `output`, `generic_body`,
`thermophysical`, `gpu` and `advanced`, plus the optional `engine_env` block.
Every section is optional; absent keys take the defaults below.

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
| `geothermal_heat_flow` | float | -1.0 | Basal heat flow in W m<sup>-2</sup> (-1 = body default). Live on **every** body since PTM-161 was fixed (`2e06a70`): forwarded to the solvers as `ovr_Qb`, announced as `[Thermo override] applied at solver: Qb …`, and a positive value arms the property-override tripwire that aborts the run if it fails to reach one. It was silently discarded on built-in bodies before that commit — [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) section 4 has the old behavior and the re-measurement |
| `heat_flow_mask_file` | string | `""` | Face-index file for locally elevated basal flux |
| `heat_flow_mask_value` | float | 1.0 | Basal flux on the masked faces (W m<sup>-2</sup>) |
| `integration_days` | float | -1.0 | Statistics-integration window (-1 = the output window) |
| `integration_steps` | int | -1 | Same window, expressed in steps |
| `integration_offset_days` | float | 0.0 | Offset of the integration window |
| `integration_offset_steps` | int | 0 | Same offset, expressed in steps |
| `initial_state_file` | string | `""` | Load a per-face `T(z)` checkpoint and skip equilibration (see [Thermal-state checkpoints](#thermal-state-checkpoints)) |
| `final_state_file` | string | `""` | Write a per-face `T(z)` checkpoint at the end of the run |
| `checkpoint_lookahead_steps` | int | 0 | Extra output steps integrated past the export window, so `final_state_file` is the state at the NEXT window's first epoch (see [Thermal-state checkpoints](#thermal-state-checkpoints)) |
| `state_restore_allow_mismatch` | list | `[]` | Per-field opt-out for the checkpoint identity guards. Names one or more of `depth_grid`, `mesh`, `config`, `body`, `grid_m`, `grid_n`, `grid_b`, `refgrid_H`, `epoch` and downgrades that clause from a refusal to a warning. There is no blanket force |
| `checkpoint_every_output_steps` | int | 0 | Mid-run checkpoint cadence (0 = chunk boundaries only) |
| `eq_indirect_source` | string | `auto` | Where the equilibration phase gets its indirect flux: `auto` estimates it from the coupling loop's first sweep, `converged` from its converged sweep (better and ~1.9× the wall clock), `off` is the pre-2026-08 behavior ([radiation-exchange.md §7.7](radiation-exchange.md)) |
| `eq_indirect_passes` | int | 1 | How many times that estimate is refreshed and the equilibration repeated. Must be ≥ 1. A second `auto` pass is that estimator's knee but the poorest value on offer — spend on `eq_indirect_source: converged` first ([radiation-exchange.md §7.7](radiation-exchange.md)) |
| `eq_indirect_flux_file` | string | `""` | Explicit per-face `'QIND'` mean-indirect file; overrides `eq_indirect_source` |
| `indirect_mean_out_file` | string | `""` | Write the converged per-face output-window mean indirect flux as a `'QIND'` file |
| `compute_ice_stability` | bool | false | Compute the subsurface ice-stability depth |
| `enable_shadow_flux` | bool | false | Sub-grid crater shadowing correction |
| `crater_dd_ratio` | float | 8.0 | Diameter-to-depth ratio for that correction |
| `spinup_seed` | string | `physical` | The initial condition. `physical` seeds every face from its own cycle-mean conductive steady profile, so a permanently shadowed column starts on the geothermal floor rather than 42–47 K above it. `legacy` restores each path's pre-2026-08-11 seed **and** the analytic pre-conditioner, i.e. exactly what produced every result in this repository before that date — use it to reproduce an old baseline. An unrecognized name is fatal at startup, never a silent fallback |
| `seed_anchor` | string | `lumped` | Which surface anchor the shared seed hangs its conductive profile from. Read **only** under `spinup_seed: physical`. `lumped` anchors at the cycle mean; `warm` uses the fourth-moment anchor, which by Jensen is ≥ the mean (+45 K on the lit population of the shipped lunar 80N crater) and is byte-identical to the pre-2026-08-12 seed |
| `seed_anchor_refuse_two_timescale` | bool | false | Refuse the lumped anchor on a face carrying both diurnal and seasonal power, where one conductance cannot cover both. Off by default because refusing substitutes `warm`, the estimator with the +45 K bias. Detection and reporting are always on regardless |
| `analytic_seed_cycles` | float | -1.0 | Diurnal cycles of the CPU analytic clear-sky pre-conditioner: `cos(lat)·cos(h)` insolation with no topography, no shadowing, no IR and no indirect flux. It is a **forcing, not a seed**, it is not counted by `equilibration_orbits`, and the GPU has no analog. -1 follows `spinup_seed` (physical → 0, legacy → the historic count: Moon 24, Europa 2, Mercury 4); 0 is off; a positive value is used exactly and announced |
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
- **The write is atomic.** The engine writes a temporary file beside the
  destination, `fsync`s it and renames it into place, so at every instant the
  destination is either the previous complete checkpoint or the new one — never
  a prefix of either.


#### The v2 identity trailer

A checkpoint written by this engine version is **PTMT v2**: the v1 header and
body are unchanged byte for byte, and an *identity trailer* is appended after
them. Every state banked before v2 existed still loads.

The trailer records what the column was integrated with:

| Field | What it is for |
|---|---|
| `z[n_layers]` | the depth grid, so a restore onto a **different** grid with the same layer count is caught |
| `et_seconds` | the SPICE ephemeris time the column is valid at |
| `output_cadence_s` | the writing run's output cadence |
| `refgrid_H`, `grid_m/n/b` | the grid parameters, named individually so a mismatch report is actionable |
| `mesh_geomhash` | a face-area hash of the mesh |
| `config_identity_hash` | the resolved thermophysical scalars and grid parameters — see below for what it does **not** cover |
| `body_name`, `engine_git_sha` | provenance |
| `flags` | pre- or post-radiation-coupling, whether every face was patched, and whether the writing run had `coupling_select_dt > 0` |
| `crc32` | a checksum over the **whole file**, so a length-correct but zero-filled cloud-sync placeholder cannot pass as a checkpoint |

`ptm_state_info <file>` prints all of it, verifies the checksum, and — given
`--body`/`--m`/`--n`/`--b` — cross-checks the depth grid the file *records*
against the grid those settings would *build*.

**What the physics hash does not cover.** Per-face maps (albedo, thermal
inertia, `H`) and a per-face `Qb` mask change the physics face by face and are
outside the hash; so are the view factors. Two runs that differ only in a loaded
map can produce the same hash. `src/cpp/thermal/state_identity.h` is the single
list of what is hashed, and the engine prints that boundary in the log whenever
a checkpoint is in play.

**`W(0)` is reserved and always written as NaN.** It is an input to a
warm-start admissibility rule that is not shipped. Computing it needs a
reduction over the output-window surface time series, which exists at the moment
of the write on some solver paths and not on others, so populating it where it
happens to be available would make a present field mean different things
depending on which solver ran. `ptm_state_info` reports it as "not recorded".


#### What a restore refuses

A restore compares the trailer against the live run. Each clause refuses
independently and names itself:

| Clause | Behavior |
|---|---|
| `depth_grid` | the recorded `z[]` differs elementwise (relative 1e-12) → **refuse**, naming the first node |
| `mesh` | `mesh_geomhash` differs → **refuse** |
| `config` | `config_identity_hash` differs → **refuse** |
| `body` | a different body → **refuse** |
| `grid_m` / `grid_n` / `grid_b` / `refgrid_H` | → **refuse**, naming the knob |
| `epoch` | see below |
| batch-completion partial | → **refuse**, and this one is not overridable |

`thermal.state_restore_allow_mismatch` takes a **list of those field names** and
downgrades each named clause to a loud warning. There is deliberately no
blanket force: turning every guard off at once is the warn-instead-of-refuse
policy under a different name, and warning instead of refusing is what made the
original checkpoint defect possible.

**The epoch clause is the one that separates a hand-off from a warm start.**

- The checkpoint's epoch equals this run's first integrated epoch → the states
  **tile** the timeline. This is the chunked workflow, and it passes silently.
- The checkpoint is **behind** by up to one output cadence → a hand-off written
  without `checkpoint_lookahead_steps: 1`, so the restored column is one step
  stale. The run **warns**, printing both epochs and the cadence, and continues.
- The checkpoint is **behind** by more than that → the banked column is a state
  that never existed at this run's epoch. The run **refuses**.
- The checkpoint is **ahead** of this run's first epoch → the two windows
  **overlap**: the run would re-integrate a span the column has already passed.
  The run **refuses**, with its own message, because the advice for a stale
  hand-off would make an overlap worse.

That last case is not a formality. A cross-season warm start measured on one
A3GT site moved `temperature_min` by 5.75 K rms and 49.9 K maximum, and shifted
the ice-stability depth product by 19.3 percentage points — and no check in the
QA battery detected it. Chaining is admissible only where the states tile the
timeline, which is what `scripts/run_chunked_simulation.py` does within a
window; across windows or seasons it is not.

**A v1 file still loads.** It cannot answer any of the questions above, so the
restore prints one line per unverified quantity, naming each, instead of
proceeding in silence. Re-bank the state on the current engine to get real
guards.

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
| `exchange_selection` | string | `all` | Which faces keep the computed indirect flux. `all` (default) — every face keeps it, and the result does not depend on how the run was split into chunks. `shadow` — only faces shadowed at least `min_shadow_fraction` of the time keep it; provided to reproduce runs published before the default changed, and best paired with `coupling_mode: legacy` |
| `min_shadow_fraction` | float | 0.3 | The shadow-fraction cut used by `exchange_selection: shadow`. **Read only under that setting**, so it is inert in a default configuration. The shadow fraction is measured over the run's *own* illumination series, so a warm-started chunk (which has no equilibration samples) reclassifies faces and zeroes their indirect flux |
| `view_factor_file` | string | `""` | Save/load path for the view-factor matrix; a relative path resolves against the configuration file's directory |
| `use_hmatrix` | bool | false | Hierarchical-matrix compression for large meshes |
| `use_monte_carlo_vf` | bool | false | Monte Carlo ray-gather view-factor builder |
| `mc_rays_per_face` | int | 100000 | Rays cast per face by that builder; read only when it is selected |
| `nearfield_quadrature_order` | int | 3 | Order of the symmetric double-area quadrature for near-field pairs. 3 is the validated production value. Order 1 is the centroid estimator with no near-field treatment at all and reads **+51.7 %** on the touching-pair closed form (order 3 reads +18.5 %); the engine calls it "a diagnostic setting, not a production one" and says so at startup |
| `coupling_mode` | string | `bounded` | How the thermal–radiation coupling loop decides it has converged. `bounded` (default) — stop once the largest per-face temperature change falls below `coupling_temp_threshold`, which bounds the residual on *every* face; this is the only data-driven exit, and `coupling_flux_threshold` is **not** part of it. `fixed` — early exit disabled entirely: every window runs exactly `max_coupling_iterations` sweeps, so the answer cannot depend on how the window was chunked. `legacy` — stop once only `coupling_face_fraction` of faces are still changing; no error bound, and it made a coupled run's answer depend on the chunking. It is the only mode that reads `coupling_face_fraction`. Compared **exactly**: `Bounded` is a startup error |
| `max_coupling_iterations` | int | 16 | Picard iteration cap (raised from 10 on 2026-08-15; see radiation-exchange.md §7.5) |
| `coupling_temp_threshold` | float | 0.5 | Temperature convergence threshold (K) — the loop's only data-driven exit, under `coupling_mode: bounded` |
| `coupling_flux_threshold` | float | 0.1 | Indirect-flux cut used to **select faces** for another thermal solve, in W/m². It is not a convergence criterion, and it is read only under the non-default `coupling_select_mode: legacy`; the default physics-based selection ignores it |
| `beaming` | bool | false | Smith (1967) anisotropic beamed thermal emission. **Standard for lunar polar SITE/REGIONAL configurations**; OFF on sub-cm meshes — see below |
| `rms_slope_deg` | float | 0.0 | RMS sub-facet slope $w$ in degrees; the beaming knob. Inert unless `beaming: true` |
| `use_gpu` | bool | false | OpenCL radiosity solver |
| `verbose` | bool | false | Detailed radiation progress |

### Beamed emission: `beaming` + `rms_slope_deg`

Rough regolith beams thermal IR toward the Sun, because the Sun-facing micro-facets are the hot
ones. In a polar crater that redistributes sunlit-wall emission onto the shadowed floor, and it is
worth several kelvin there. The physics, the validated result and the full cost table are in
[radiation-exchange.md §8](radiation-exchange.md); this is the configuration surface.

Two keys, both required:

```yaml
radiation:
  beaming: true            # THE SWITCH. Default false; nothing beams without it.
  rms_slope_deg: 32.5      # THE KNOB. RMS sub-facet slope w, degrees.
```

- `beaming: true` with `rms_slope_deg` unset or $\le 0$ is a **startup error**. The engine will not
  quietly run Lambertian under a config that asked for beaming.
- `beaming: false` with a nonzero `rms_slope_deg` runs Lambertian and prints one line saying the
  slope is inert. Keeping the value in the file as documentation is fine.
- A config written before 2026-08-25 has no `beaming` key, and back then `rms_slope_deg > 0` *was*
  the switch. Those configs still beam, and print a `[config] NOTE` asking for the explicit key —
  so no existing run changes physics under you. New configs should set both keys.

**Choosing $w$.** It is a surface property, not a fit parameter. Published lunar values span
20–37°: 32.5° (Jhoti et al. 2024, Table 1 — lunar highlands; the validated polar value and the
recommended default), ~20° (Bandfield et al. 2015, Diviner global), ~37° (Rubanenko et al. 2020).

**When it applies — compare two lengths, not two mission names.** Beaming is a *sub-facet* model:
it stands in for roughness the mesh does not resolve, at the scale where surface elements become
thermally isolated from one another — **millimeters to centimeters** on lunar regolith. Paul's
ruling of 2026-08-25:

> "The Moon3D mesh resolution is comparable to the roughness scale (millimeters to centimeters)
> that the beaming is intended to capture, so the beaming should be OFF for small local meshes
> like this."

So the rule is: **turn beaming ON only when the facet scale is much larger than ~cm.** At or below
that scale the mesh already resolves the anisothermality and beaming double-counts it. Measured
facet scales, $\sqrt{\text{mean face area}}$:

| Mesh | $\sqrt{\bar{A}}$ | beaming |
|---|---|---|
| `sunlit_radcore_r50` (50 km core, 140k faces) | 237 m | **ON** |
| `haworth_radcore_r32c` (32 km core, 100k faces) | 181 m | **ON** |
| A3GT lunar polar site A (3 km core, 2.4 M faces) | 3.46 m | **ON** |
| `moon3d` 509k (a ~1.6 m simulated crater) | 1.96 mm | **OFF** |

**Policy.** Beaming is standard for **site and regional** lunar polar configurations — meters and
up. It is deliberately *not* enabled in `examples/moon3d_native_yearlong/*.yaml` (sub-cm facets,
per the ruling above), in `examples/validation/baseline_crater80N_*.yaml` (those are the
byte-identical regression baselines) or in the `tutorial04_crater80N_*` teaching configs, and it
must not be carried to Europa or asteroid configs — $w = 32.5°$ is a lunar regolith number
validated on lunar polar-crater geometry.

**Cost.** About **+22% wall time and 5–6.5 GiB of host RAM** on a 140k-face lunar polar mesh
(measured). The memory term scales with the view-factor matrix's nonzero count and reaches an
estimated 22–28 GiB on a 2.4 M-face A3GT site — measure before flipping it on a large mesh. Every
run with beaming on prints its own `[beam-cost]` breakdown.

**Two caveats worth knowing before you turn it on.** Beaming makes a run
chunk-decomposition dependent (its emitter/receiver sets are window-scoped, PTM-101 — the run
warns), and it does **not** address the known winter/margin warm biases; those are open and parked.

## flux_cubes

A passive probe that samples the flux field at fixed points in space. The block
is **top level**, not nested under `output:` — deliberately, so it cannot be
reached by the body-defaults machinery that makes `thermophysical:` decorative
on a built-in body (PTM-67). The physics, the geometry cache, the ray budget and
the outputs are all in [flux-cubes.md](flux-cubes.md); this table is the key
list.

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | false | Run the flux-cube pass. Turning it on must be byte-neutral to every mesh field |
| `cache_file` | string | `""` | Geometry cache built by `ptm_flux_cube_geom`. **Required** when enabled |
| `locations` | list[map] | — | Named locations (`name`, `lat_deg`, `lon_deg`, optional `height_m`, `edge_m`). The parser's only sequence of maps, so its per-entry fields have no dotted keys |
| `write_binary` | bool | true | Write `flux_cubes/flux_cubes.fcb` |
| `write_csv` | bool | true | Write `flux_cubes/csv/cube_XXXXX.csv` |
| `write_csv_max_cubes` | int | 0 | Cap the CSV fan-out on a long window (0 = every cube). Never drops a named location |
| `shadow_probe_rays` | int | 5 | 1 = cube center only; 5 = center plus four rim points |

The geometry keys (`spacing_m`, `height_factor`, `n_rays`, …) are **accepted and
reported inert**: they live in the cache header, and honoring a YAML copy would
create two sources of truth for one number. `warnUnreadFluxCubeKeys()` names any
key the parser does not consume.

**CLI-only for now.** There is no GUI surface for this section; whether to build
one is decision DP-7 in [design/gui-update-plan.md](design/gui-update-plan.md).

## gpu

Cross-cutting GPU **safety policy**: whether a GPU rung is allowed at all, and
which devices each pool may use. This is distinct from the per-solver
`gpu_devices` / `gpu_device_ids` / `gpu_batch_size` keys, which tune a rung that
has already been allowed. The full rationale, the platform-gating table and the
cost of each guardrail are in
[gpu-wedge-guardrails.md](gpu-wedge-guardrails.md).

| Key | Type | Default | Description |
|---|---|---|---|
| `min_vram_margin_gib` | float | 2.0 | Floor on the per-device VRAM headroom a resident radiosity CSR replica may leave behind. A rung with less is refused (see gating below). 0 = fit check only, the pre-2026-08-22 behavior |
| `placement_policy` | string | `auto` | `auto`, `shared` or `disjoint`: may the radiosity replica and the GPU thermal pool occupy the same devices? |
| `placement_margin_gib` | float | 4.0 | Margin below which `auto` separates the two pools onto disjoint devices, and below which the thin-margin advisory prints |
| `thin_margin_policy` | string | `separate` | What a margin below `min_vram_margin_gib` does: `separate` the two pools onto disjoint devices and keep the GPU rung, or `refuse` the rung outright |
| `hard_min_vram_margin_gib` | float | 0.25 | Under `separate`, the floor below which the GPU rung is refused anyway |

**Platform gating.** The failure these guard against — a wedged GPU that takes
the machine down with it — was measured on macOS with AMD (Navi21) devices.

| | macOS + AMD | every other host |
|---|---|---|
| `min_vram_margin_gib` | **enforces**: a thin rung is refused and radiosity runs on the CPU | **advisory**: prints the same message and proceeds |
| `placement_policy: auto` | separates the pools when the margin is thin | never separates; prints a note when it would have |
| `placement_policy: disjoint` | separates | separates (an explicit instruction is obeyed everywhere) |

Setting `min_vram_margin_gib` in the config (or `PTM_GPU_MIN_VRAM_MARGIN_GIB`
in the environment) makes it enforcing on **every** host: an operator who asks
for a policy gets the policy.

**What the 2.0 default does.** A site whose CSR replica fits by less than 2 GiB
is treated as thin, and by default (`thin_margin_policy: separate`) the engine
moves the radiosity replica and the GPU thermal pool onto **disjoint devices**
and keeps the GPU rung — a derived +7-10 min per A3GT-scale window. It gives up
the rung only below `hard_min_vram_margin_gib` (0.25), or on a rung that cannot
be separated. That matters because the alternative is expensive: measured on
lunar polar site B (which fit by 0.64 GiB), CPU radiosity cost 108.9 min/window against
31.4 min on the GPU pool, 8 windows each. The reason the margin needs guarding
at all is that the GPU thermal pool allocates roughly 0.77 GiB per device on
the *same* dies while the replica is resident, and the fit check does not count
it. Set `thin_margin_policy: refuse` to give up the rung instead; see
[gpu-wedge-guardrails.md](gpu-wedge-guardrails.md).

`auto` only engages when **all** of: the thermal solver is on the GPU, the two
device sets overlap, the margin is below `placement_margin_gib`, the host is
macOS + AMD, and neither side pinned explicit device ids. An explicit pin is an
operator decision and `auto` will not overrule it — it warns instead. The
single-GPU radiosity rung cannot select its device, so the separation is not
available there; the engine says so rather than sharing the die in silence.

```yaml
gpu:
  min_vram_margin_gib: 2.0
  placement_policy: auto
  placement_margin_gib: 4.0
```

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
| `H_override` | float | -1.0 | Regolith density/conductivity e-folding depth H, in meters — the scale in `rho(z) = rhod + (rhos-rhod)·exp(-z/H)` and `kc(z) = kd + (ks-kd)·exp(-z/H)`. Inert when `constant_thermal_inertia > 0` (which makes those exponentials multiply zero) and out-ranked per face by `thermal.h_parameter_map` / `thermal.thermal_inertia_map`; both cases warn |
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
block is **refused at config parse**, with `rc 1` and a message naming the three
implemented bodies and the generic-body route. Until 2026-08-26 such a name fell
through to Europa's property set and Europa's metakernel instead, and the run
completed with `rc 0` and zero absorbed flux on every face. Use `generic_body:`
for anything outside the three built-ins.)

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

## Checking a config without running it

```bash
./build/bin/ptm_thermal_sim --config my_config.yaml --validate-config
```

Parses the file, applies any command-line overrides, runs every startup check
that is knowable from the configuration alone, and exits. `rc 0` = accepted,
`rc 1` = refused, with the engine's own message on stdout/stderr. Nothing is
simulated, no output directory is created, and an existing output directory —
including its `heartbeat.txt` — is left alone.

It covers YAML syntax and structure, the `PTM_DATA_ROOT` resolution contract,
the enum values (`conductivity_model`, `coupling_mode`, `exchange_selection`,
`gpu.placement_policy`, `gpu.thin_margin_policy`), the numeric ranges
(`nearfield_quadrature_order`, `min_shadow_fraction`, the `gpu.*` margins), the
`beaming`/`rms_slope_deg` pair, and whether this build has the GPU support the
config asks for.

It does **not** cover anything that needs the mesh, the body or SPICE: mesh
loading and units, generic-body construction, SPICE kernels, property-map face
counts, the `martinez2021` density guard, view factors, and the spin-up seed
strings. The mode prints that boundary every time it runs. **`rc 0` means the
configuration is well formed, not that the run will complete.**

The GUI's *Simulation → Check Configuration* uses this and shows the engine's
text verbatim, rather than maintaining a second, weaker copy of these rules.

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
