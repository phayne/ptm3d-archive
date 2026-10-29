# Configuration Guide

This document lists the YAML configuration keys PTM3D reads, with the defaults the
parser applies when a key is absent (`src/cpp/tools/config_parser.cpp`).

> Changing a key or a default in the parser? Regenerate the Python layer's copy of
> this table as well; see [CONTRIBUTING.md](../CONTRIBUTING.md#changing-an-engine-configuration-key).

> Runtime behavior is also influenced by `PTM_*` environment variables; see
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

The output grid is half-open. A window of `timesteps` samples over
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

`start_time_offset_seconds` shifts the whole grid by a number of ephemeris
seconds. Use it, rather than a shifted UTC `start_time` string, whenever a
window boundary has to land on an exact multiple of the cadence: TDB − UTC
carries a periodic relativistic term (~1.7 ms amplitude, annual period) and
steps by a whole second at a leap second, so adding to the UTC string is not the
same shift. `scripts/run_chunked_simulation.py` uses it for every chunk.

> Changed 2026-08-12. Before that date the default (uniform-cadence) path
> computed `dt = (equilibration_days + duration_days) * 86400 / (total_steps −
> 1)`: the output cadence was set by dividing a span that included the
> equilibration prefix, by one fewer interval than there are samples. Output
> epochs on every run shifted as a result: on an 80 N lunar crater with 236
> steps over 59.06 days and one equilibration lunation, frame 0 was 1 h 59 m
> 53.7 s after the nominal start and the cadence was 0.28 % long. Products
> generated before that date carry those labels. See
> [chunked-run-equivalence.md](chunked-run-equivalence.md).

## mesh

| Key | Type | Default | Description |
|---|---|---|---|
| `resolution` | int | 5 | Icosphere subdivision level when no mesh is imported |
| `import_file` | string | `""` | Mesh file to load (PLY/OBJ/VTK) |
| `occluder_file` | string | `""` | Shadow-only occluder mesh, added to the ray-tracing scene but never solved or written; see [extended-occluder-architecture.md](extended-occluder-architecture.md) |
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
| `equilibration_orbits` | float | 1.0 | Integrated diurnal cycles of real, ephemeris-driven spin-up before the output window. The same number on the CPU scalar, CPU SIMD and GPU persistent paths; `equilibration_cycles` changes how it is reached, never the total. Under the legacy seed the CPU paths also run an analytic pre-conditioner this key does not count; see [Equilibration length](#equilibration-length-what-the-number-means) and [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md), item 2 |
| `equilibration_steps_per_day` | float | -1.0 | Explicit equilibration cadence in steps per Earth-day (-1 = unset) |
| `equilibration_samples_per_diurnal_cycle` | float | -1.0 | Target illumination samples per diurnal cycle; consulted only when the explicit cadence is unset. 24 gives about 2 K accuracy on the Moon, 48 about 1 K |
| `equilibration_cycles` | int | -1 | Replays of the equilibration forcing block used to reach `equilibration_orbits`, GPU persistent solver only (-1 = 1 replay). The block is `equilibration_orbits / equilibration_cycles` cycles long, so the integrated total does not change and only the flux RAM and illumination cost fall |
| `equilibration_target_K` | float | 0.1 | Accuracy the spin-up estimate targets, and the bar the retrospective certificate must clear; see [Did the column settle?](#did-the-column-settle-the-retrospective-certificate) |
| `equilibration_stop_on_certificate` | bool | false | End the analytic pre-conditioner when the certificate reaches `equilibration_target_K` instead of at a fixed cycle count. CPU SIMD batch path only; the other solver paths refuse to start with it set |
| `eq_extrapolation` | map | off | Pass-stride extrapolation of the equilibration state; see below |
| `compute_statistics` | bool | true | Compute and write per-face statistics |
| `use_gpu` | bool | false | OpenCL thermal solver (a CPU-only build refuses at startup) |
| `gpu_devices` | int | 1 | Devices in the thermal pool (0 = all enumerated) |
| `gpu_device_ids` | list[int] | [] | Explicit device indices; overrides `gpu_devices` while non-empty. Must be a list; a scalar is refused, because it would drop the pin rather than move it ([why](#keys-the-engine-reads-as-lists)) |
| `gpu_batch_size` | int | 0 | Faces per GPU batch (0 = automatic cap) |
| `gpu_inter_batch_pause_ms` | int | 0 | Pause between consecutive batches on a device |
| `auto_calibrate_gpu` | bool | true | Probe the host once for a dispatch ceiling when no machine profile exists |
| `thermal_inertia_map` | string | `""` | Per-face `float32` thermal-inertia map |
| `h_parameter_map` | string | `""` | Per-face `float32` H-parameter map (mutually exclusive with the above) |
| `use_default_property_map` | bool | false | Auto-discover the bundled property map for the body |
| `property_maps_directory` | string | `"data/property_maps"` | Where to look for property maps |
| `geothermal_heat_flow` | float | -1.0 | Basal heat flow in W m<sup>-2</sup> (-1 = body default: 0.018 Moon, 0.030 Europa, 0.020 Mercury). Values > 0 are applied by every solver; the run confirms with `[Thermo override] applied at solver: Qb ...`. Note that `0` means "body default", not zero |
| `heat_flow_mask_file` | string | `""` | Face-index file for locally elevated basal flux |
| `heat_flow_mask_value` | float | 1.0 | Basal flux on the masked faces (W m<sup>-2</sup>) |
| `integration_days` | float | -1.0 | Statistics-integration window (-1 = the output window) |
| `integration_steps` | int | -1 | Same window, expressed in steps |
| `integration_offset_days` | float | 0.0 | Offset of the integration window |
| `integration_offset_steps` | int | 0 | Same offset, expressed in steps |
| `initial_state_file` | string | `""` | Load a per-face `T(z)` checkpoint and skip equilibration (see [Thermal-state checkpoints](#thermal-state-checkpoints)) |
| `final_state_file` | string | `""` | Write a per-face `T(z)` checkpoint at the end of the run |
| `checkpoint_lookahead_steps` | int | 0 | Extra output steps integrated past the export window, so `final_state_file` is the state at the next window's first epoch (see [Thermal-state checkpoints](#thermal-state-checkpoints)) |
| `state_restore_allow_mismatch` | list | `[]` | Per-field opt-out for the checkpoint identity guards. Names one or more of `depth_grid`, `mesh`, `config`, `body`, `grid_m`, `grid_n`, `grid_b`, `refgrid_H`, `epoch` and downgrades that clause from a refusal to a warning. There is no blanket force |
| `checkpoint_every_output_steps` | int | 0 | Mid-run checkpoint cadence (0 = chunk boundaries only). Architecturally inactive on the GPU path; see [Batch-completion checkpointing](#batch-completion-checkpointing) |
| `batch_checkpoint` | bool | false | Bank each *completed face batch* so an interrupted run resumes from it instead of re-integrating the window. CPU SIMD-batch solver only; every other path refuses. See [Batch-completion checkpointing](#batch-completion-checkpointing) |
| `batch_checkpoint_dir` | string | `""` | Where the partials go. Empty = `<output.directory>/_thermal_state/partial` |
| `batch_checkpoint_max_gib` | float | 64.0 | Refuse to start if the projected partial set exceeds this. 0 disables the check |
| `batch_checkpoint_flush_mib` | float | 0.0 | How much completed work to accumulate before writing a partial. 0 derives it as one sixteenth of the projected set, clamped to [1, 64] MiB |
| `eq_indirect_source` | string | `auto` | Where the equilibration phase gets its indirect flux: `auto` estimates it from the coupling loop's first sweep, `converged` from its converged sweep (better and ~1.9× the wall clock), `off` is the pre-2026-08 behavior ([radiation-exchange.md §7.7](radiation-exchange.md)) |
| `eq_indirect_passes` | int | 1 | How many times that estimate is refreshed and the equilibration repeated. Must be ≥ 1. A second `auto` pass is that estimator's knee but the poorest value on offer; use `eq_indirect_source: converged` first ([radiation-exchange.md §7.7](radiation-exchange.md)) |
| `eq_indirect_flux_file` | string | `""` | Explicit per-face `'QIND'` mean-indirect file; overrides `eq_indirect_source` |
| `indirect_mean_out_file` | string | `""` | Write the converged per-face output-window mean indirect flux as a `'QIND'` file |
| `compute_ice_stability` | bool | false | Compute the subsurface ice-stability depth |
| `enable_shadow_flux` | bool | false | Sub-grid crater shadowing correction |
| `crater_dd_ratio` | float | 8.0 | Diameter-to-depth ratio for that correction |
| `spinup_seed` | string | `physical` | The initial condition. `physical` seeds every face from its own cycle-mean conductive steady profile, so a permanently shadowed column starts on the geothermal floor rather than 42–47 K above it. `legacy` restores each path's pre-2026-08-11 seed and the analytic pre-conditioner, i.e. exactly what produced every result in this repository before that date. Use it to reproduce an old baseline. An unrecognized name is fatal at startup, never a silent fallback |
| `seed_anchor` | string | `lumped` | Which surface anchor the shared seed hangs its conductive profile from. Read only under `spinup_seed: physical`. `lumped` anchors at the cycle mean; `warm` uses the fourth-moment anchor, which by Jensen is ≥ the mean (+45 K on the lit population of the shipped lunar 80N crater) and is byte-identical to the pre-2026-08-12 seed |
| `seed_anchor_refuse_two_timescale` | bool | false | Refuse the lumped anchor on a face carrying both diurnal and seasonal power, where one conductance cannot cover both. Off by default because refusing substitutes `warm`, the estimator with the +45 K bias. Detection and reporting are always on regardless |
| `analytic_seed_cycles` | float | -1.0 | Diurnal cycles of the CPU analytic clear-sky pre-conditioner: `cos(lat)·cos(h)` insolation with no topography, no shadowing, no IR and no indirect flux. It is a forcing, not a seed; it is not counted by `equilibration_orbits`, and the GPU has no analog. -1 follows `spinup_seed` (physical gives 0, legacy gives the historic count: Moon 24, Europa 2, Mercury 4); 0 is off; a positive value is used exactly and announced |

The checkpoint files are used by the chunked-run workflow
(`scripts/run_chunked_simulation.py` and `ptm_merge_chunks`).


### Thermal-state checkpoints

`thermal.initial_state_file` and `thermal.final_state_file` hand a whole-mesh
per-face `T(z)` column between runs. Every whole-mesh solver path reads and
writes them (the GPU persistent solver, the CPU SIMD batch solver, the CPU
scalar solver, and the streamed CPU path) in the same format
(`uint32` magic `0x50544D54` `'PTMT'`, `uint32 n_faces`, `uint32 n_layers`, then
`n_layers × n_faces` `float32` in layer-major order). A checkpoint written by
either backend restores on the other.

Three properties matter before relying on one.

- It refuses; it does not degrade. A missing file, a wrong magic number, a
  truncated body, a face- or layer-count mismatch, or a destination that cannot
  be opened for writing aborts the run with a message naming the reason. It
  never falls back to a cold start, because a silently cold-started chunk
  reports spin-up as physics.
- The file is `float32` on every path. The CPU solvers carry `T` in
  `float64`, so writing a checkpoint rounds each value once, at the write.
  Restoring is exact. A CPU-to-file-to-CPU round trip therefore reproduces the
  file bit-for-bit and the in-memory column to `float32` precision.
- The write is atomic. The engine writes a temporary file beside the
  destination, `fsync`s it and renames it into place, so at every instant the
  destination is either the previous complete checkpoint or the new one, never
  a prefix of either.


#### The v2 identity trailer

A checkpoint written by this engine version is PTMT v2: the v1 header and
body are unchanged byte for byte, and an *identity trailer* is appended after
them. Every state banked before v2 existed still loads.

The trailer records what the column was integrated with:

| Field | What it is for |
|---|---|
| `z[n_layers]` | the depth grid, so a restore onto a different grid with the same layer count is caught |
| `et_seconds` | the SPICE ephemeris time the column is valid at |
| `output_cadence_s` | the writing run's output cadence |
| `refgrid_H`, `grid_m/n/b` | the grid parameters, named individually so a mismatch report is actionable |
| `mesh_geomhash` | a face-area hash of the mesh |
| `config_identity_hash` | the resolved thermophysical scalars and grid parameters (see below for what it does not cover) |
| `body_name`, `engine_git_sha` | provenance |
| `flags` | pre- or post-radiation-coupling, whether every face was patched, and whether the writing run had `coupling_select_dt > 0` |
| `crc32` | a checksum over the whole file, so a length-correct but zero-filled cloud-sync placeholder cannot pass as a checkpoint |

`ptm_state_info <file>` prints all of it, verifies the checksum, and, given
`--body`/`--m`/`--n`/`--b`, cross-checks the depth grid the file *records*
against the grid those settings would *build*.

The physics hash does not cover everything. Per-face maps (albedo, thermal
inertia, `H`) and a per-face `Qb` mask change the physics face by face and are
outside the hash; so are the view factors. Two runs that differ only in a loaded
map can produce the same hash. `src/cpp/thermal/state_identity.h` is the single
list of what is hashed, and the engine prints that boundary in the log whenever
a checkpoint is in play.

`W(0)` is reserved and always written as NaN. It is an input to a
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
| `depth_grid` | the recorded `z[]` differs elementwise (relative 1e-12): refuse, naming the first node |
| `mesh` | `mesh_geomhash` differs: refuse |
| `config` | `config_identity_hash` differs: refuse |
| `body` | a different body: refuse |
| `grid_m` / `grid_n` / `grid_b` / `refgrid_H` | refuse, naming the knob |
| `epoch` | see below |
| batch-completion partial | refuse, and this one is not overridable |

`thermal.state_restore_allow_mismatch` takes a list of those field names and
downgrades each named clause to a loud warning. There is deliberately no
blanket force: turning every guard off at once is the warn-instead-of-refuse
policy under a different name, and warning instead of refusing is what made the
original checkpoint defect possible.

The epoch clause is the one that separates a hand-off from a warm start.

- The checkpoint's epoch equals this run's first integrated epoch: the states
  tile the timeline. This is the chunked workflow, and it passes silently.
- The checkpoint is behind by up to one output cadence: a hand-off written
  without `checkpoint_lookahead_steps: 1`, so the restored column is one step
  stale. The run warns, printing both epochs and the cadence, and continues.
- The checkpoint is behind by more than that: the banked column is a state
  that never existed at this run's epoch. The run refuses.
- The checkpoint is ahead of this run's first epoch: the two windows
  overlap; the run would re-integrate a span the column has already passed.
  The run refuses, with its own message, because the advice for a stale
  hand-off would make an overlap worse.

That last case is not a formality. A cross-season warm start measured on one
lunar south-polar site moved `temperature_min` by 5.75 K rms and 49.9 K maximum, and shifted
the ice-stability depth product by 19.3 percentage points, and no check in the
QA battery detected it. Chaining is admissible only where the states tile the
timeline, which is what `scripts/run_chunked_simulation.py` does within a
window; across windows or seasons it is not.

A v1 file still loads. It cannot answer any of the questions above, so the
restore prints one line per unverified quantity, naming each, instead of
proceeding in silence. Re-bank the state on the current engine to get real
guards.

Setting `initial_state_file` forces `equilibration_orbits` to 0 and disables the
analytic clear-sky pre-conditioner: a restored column is a continuation, and
spinning it up again would discard it. Both are announced in the log.

The solver integrates
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
un-coupled column on the faces the re-run touched. The run says so.


### Batch-completion checkpointing

`thermal.batch_checkpoint` bounds what an interrupted run loses. It is off by
default, and byte identity with it off is its acceptance test.

It is not "checkpoint every N steps." The thermal solvers are
*batch-outer / time-inner*: each face batch runs all timesteps, then the
next batch starts. There is therefore no wall-clock moment at which every face
shares an output step, and a whole-mesh dump taken mid-run would bank faces that
have not been solved. That is why `thermal.checkpoint_every_output_steps` warns
instead of working, and it is not a defect to be fixed here: per-output-step
checkpointing belongs to the CPU-first, time-outer solver on the portability
roadmap.

What a batch-outer solver *can* offer is the inverse. At a batch boundary a
well-defined set of faces is finished for the entire window, so that is what
gets banked: a self-describing partial (`PTMB`) per flush, carrying each
completed batch's surface time series, its end-of-window `T(z)` column and its
slice of the depth-map accumulator. On restart the engine validates the partials
it finds, restores those batches and integrates only the rest.

A partial carries the surface time series, not just the
column, because every statistic and every per-step product is a reduction of it.
Per face that is `n_timesteps × 8 B` plus `n_layers × 4 B`; on a 2,397,890-face
mesh with 1440 output steps the partial set is about 26 GiB per window, not
the ~210 MiB a column-only dump would take. The engine prints the exact
projection at startup and refuses above `batch_checkpoint_max_gib`.

Some losses are not bounded. A run that dies *inside* a batch loses that batch's
whole time integration; that is the granularity, by construction. And a resume
repeats the illumination and radiation setup; only the thermal integration is
banked. For finer blast-radius bounding the answer remains the runner's
process-level chunk split (`scripts/run_chunked_simulation.py`), which is
already built and already equivalent to a single shot.

The feature is implemented on the CPU SIMD-batch
whole-mesh solver only. The GPU persistent solver, the CPU scalar solver and any
radiation-coupled run refuse to start with the key set, rather than running
with the checkpoint quietly inert:

| Combination | Why it refuses |
|---|---|
| `thermal.use_gpu: true` | Not implemented there. The acceptance test for this feature is an abrupt kill mid-run, and killing a process with OpenCL queues in flight is the documented compute-wedge class on this workstation; a path whose resume equivalence cannot be crash-tested does not ship with the switch enabled |
| `--no-simd-batch` | The CPU scalar solver's depth grid is per face, so one partial identity cannot describe the set |
| `radiation.enabled: true` | The whole-mesh driver is entered more than once, and a partial set surviving across entries would let the second pass restore the first pass's results and call the window finished |

On resume, every partial carries the run's depth grid, epoch,
output cadence, mesh geometry hash, configuration identity hash, body and engine
SHA, plus a CRC32 over the whole file. Any mismatch refuses and names the field.
It also carries an **input digest**, a checksum of the exact bytes that batch
integrated (its absorbed-flux rows, its seeded column, its per-face `Q_b`). That
is the guard that catches what the configuration hash cannot: a different albedo
map, different view factors, or radiation on instead of off all leave
`config_identity_hash` unchanged and all change the digest.

Completeness is computed from the partials that validated, never taken from
`manifest.json`. The manifest is an index: if it names a file that is missing or
does not validate, the resume refuses; if it is merely behind the files on disk
(a flush landed, the manifest rewrite did not) that is the normal crash tail and
is reported. A partial can never be loaded as `thermal.initial_state_file`; it
is a different format and fails at the magic.

The crash-resume test is `scripts/batch_checkpoint_crash_gate.py`, which
SIGKILLs the engine at several points, resumes, and requires the finished
outputs to be byte-identical to an uninterrupted control.


### Equilibration length: what the number means

`thermal.equilibration_orbits` is the integrated number of diurnal cycles of
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
509,337-face lunar mesh. It is a trade-off: *N* replays of a
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

`equilibration_orbits` does not count the pre-conditioner. Under `spinup_seed: legacy` the CPU scalar and
SIMD paths first run an analytic clear-sky pre-conditioner (`NYEARSEQ * round(year/day)` cycles:
24 on the Moon, 2 on Europa, 4 on Mercury) with flat, unshadowed insolation. Under the default
`spinup_seed: physical` it is off (`analytic_seed_cycles` resolves to 0) unless requested, and the
banner prints its length either way.

Before 2026-08-11, `equilibration_orbits` was the *block* length and the GPU
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
they have failed. On the banked 8-pass lunar state (bit-frozen across 100
further lunations), the certificate still reports 0.26 K against a 0.1 K
target.

Which `h` depends on the product:

| Observable | `h` |
|---|---|
| diurnal maximum | `4 eps sigma T_max^3` |
| diurnal minimum | `4 eps sigma T_min^3` |
| diurnal mean | the harmonic cycle mean of `4 eps sigma T0(t)^3` |
| deep products | `E_deep = |r_c| tau1 / C_below` (unbiased, not conservative: pointwise 0.91–1.09; reported, never used to pass) |

The mean uses the harmonic cycle mean and not the arithmetic one. The surface
displacement caused by a flux residual is largest exactly where the radiative
stiffness is smallest, so the two are anti-correlated and the arithmetic mean
understates the error (by 8.7x, measured on a 25 %/75 % duty cycle between
350 K and 100 K). The minimum is in the table for the same reason: it is the
slackest instant of the cycle and therefore the most flux-sensitive surface
observable. Gating a lunar spin-up on the mean alone stopped it 13.6 cycles
early and left `temperature_min` 0.69 K from its full-length answer while
`temperature_max` moved 0.03 K; gating on all three left it at 0.06 K.

After a run, `build/bin/ptm_eq_certify` evaluates the certificate against a
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

During a run, the certificate is used through `thermal.equilibration_stop_on_certificate` (default false).
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

On a 509,337-face lunar polar mesh at
`order_p: 4` with five passes, the extrapolated state was 55x closer (sunlit
faces) and 64x closer (seasonal-terminator faces) to a 200-lunation reference
than four passes of plain cycling, for 1.18x the wall-clock of four passes. The
benefit accrues to faces that carry energy.

About 70 percent of that mesh receives no flux during equilibration, and those
faces gain nothing from extrapolation, but not because they have converged.
On the GPU path their deep columns stop changing between passes while still
about 0.36 K short of the steady state, so extrapolating a sequence that has
already stopped moving has nothing to work with. Do not read "unchanged between
passes" as "converged"; see
[KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) sections 1 and 3.

There is one known limitation. The fit assigns a weight to the first snapshot in its
window, which is the initial guess. For a depth layer that converged early while
shallower layers were still evolving, that weight can pull the layer back toward
the initial guess. The displacement is bounded by

    displacement  <=  |gamma_0| * (initial-guess error at that depth)

with `gamma_0` the weight on the first snapshot, measured at about 0.09, so
roughly ten percent of the initial-guess error, not an amplification of it. The
`hold_K` test suppresses this by holding any layer whose last-pass increment is
below the threshold, and the run reports how many windows remain susceptible.
On the mesh above, 73 of 509,337 faces retained a displacement, the largest
0.47 K at 1 m depth, all on faces receiving no flux during equilibration.
Raising `hold_K` holds more layers and trades extrapolation speed-up for
reliability; the layers it holds simply converge at the ordinary rate.

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
| `gpu_device_ids` | list[int] | [] | Explicit device indices; must be a list (a scalar is refused) ([why](#keys-the-engine-reads-as-lists)) |
| `gpu_inter_batch_pause_ms` | int | 0 | Pause between consecutive batches on a device |
| `bvh_cache_file` | string | `""` | BVH cache path (empty = auto under the output directory; `off` disables) |

## radiation

Radiation exchange is configured from YAML as well as from the command line; the
command line wins. The full key list, with the physics behind each, is in
[radiation-exchange.md §9](radiation-exchange.md). The most commonly used keys:

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | false | Enable inter-face radiation exchange |
| `band_separated` | bool | true | Band-separated visible + thermal-IR radiosity; false = single gray band |
| `exchange_selection` | string | `all` | Which faces keep the computed indirect flux. `all` (default): every face keeps it, and the result does not depend on how the run was split into chunks. `shadow`: only faces shadowed at least `min_shadow_fraction` of the time keep it; provided to reproduce runs published before the default changed, and best paired with `coupling_mode: legacy` |
| `min_shadow_fraction` | float | 0.3 | The shadow-fraction cut used by `exchange_selection: shadow`. Read only under that setting, so it is inert in a default configuration. The shadow fraction is measured over the run's *own* illumination series, so a warm-started chunk (which has no equilibration samples) reclassifies faces and zeroes their indirect flux |
| `view_factor_file` | string | `""` | Save/load path for the view-factor matrix; a relative path resolves against the configuration file's directory |
| `use_hmatrix` | bool | false | Hierarchical-matrix compression for large meshes |
| `use_monte_carlo_vf` | bool | false | Monte Carlo ray-gather view-factor builder |
| `mc_rays_per_face` | int | 100000 | Rays cast per face by that builder; read only when it is selected |
| `nearfield_quadrature_order` | int | 3 | Order of the symmetric double-area quadrature for near-field pairs. 3 is the validated production value. Order 1 is the centroid estimator with no near-field treatment at all and reads +51.7 % on the touching-pair closed form (order 3 reads +18.5 %); the engine calls it "a diagnostic setting, not a production one" and says so at startup |
| `coupling_mode` | string | `bounded` | How the thermal–radiation coupling loop decides it has converged. `bounded` (default): stop once the largest per-face temperature change falls below `coupling_temp_threshold`, which bounds the residual on *every* face; this is the only data-driven exit, and `coupling_flux_threshold` is not part of it. `fixed`: early exit disabled entirely; every window runs exactly `max_coupling_iterations` sweeps, so the answer cannot depend on how the window was chunked. `legacy`: stop once only `coupling_face_fraction` of faces are still changing; no error bound, and it made a coupled run's answer depend on the chunking. It is the only mode that reads `coupling_face_fraction`. Compared exactly: `Bounded` is a startup error |
| `max_coupling_iterations` | int | 16 | Picard iteration cap (raised from 10 on 2026-08-15; see radiation-exchange.md §7.5) |
| `coupling_temp_threshold` | float | 0.5 | Temperature convergence threshold (K); the loop's only data-driven exit, under `coupling_mode: bounded` |
| `coupling_flux_threshold` | float | 0.1 | Indirect-flux cut used to select faces for another thermal solve, in W/m². It is not a convergence criterion, and it is read only under the non-default `coupling_select_mode: legacy`; the default physics-based selection ignores it |
| `beaming` | bool | false | Smith (1967) anisotropic beamed thermal emission. Standard for lunar polar site/regional configurations; off on sub-cm meshes (see below) |
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

- `beaming: true` with `rms_slope_deg` unset or $\le 0$ is a startup error. The engine will not
  quietly run Lambertian under a config that asked for beaming.
- `beaming: false` with a nonzero `rms_slope_deg` runs Lambertian and prints one line saying the
  slope is inert. Keeping the value in the file as documentation is fine.
- A config written before 2026-08-25 has no `beaming` key, and back then `rms_slope_deg > 0` *was*
  the switch. Those configs still beam, and print a `[config] NOTE` asking for the explicit key,
  so no existing run changes physics. New configs should set both keys.

The slope $w$ is a surface property, not a fit parameter. Published lunar values span
20–37°: 32.5° (Jhoti et al. 2024, Table 1, lunar highlands; the validated polar value and the
recommended default), ~20° (Bandfield et al. 2015, Diviner global), ~37° (Rubanenko et al. 2020).

Whether beaming applies is decided by comparing two lengths, not two mission names. Beaming is a *sub-facet* model:
it stands in for roughness the mesh does not resolve, at the scale where surface elements become
thermally isolated from one another: millimeters to centimeters on lunar regolith.

The rule is therefore to turn beaming on only when the facet scale is much larger than ~cm. At or below
that scale the mesh already resolves the anisothermality and beaming double-counts it. Measured
facet scales, $\sqrt{\text{mean face area}}$:

| Mesh | $\sqrt{\bar{A}}$ | beaming |
|---|---|---|
| `sunlit_radcore_r50` (50 km core, 140k faces) | 237 m | on |
| `haworth_radcore_r32c` (32 km core, 100k faces) | 181 m | on |
| a 3 km lunar polar site core (2.4 M faces) | 3.46 m | on |
| Moon3D testbed scan (509,337 faces) | 1.96 mm | off |

Beaming is standard for site and regional lunar polar configurations (meters and
up). It is deliberately *not* enabled in `examples/moon3d_native_yearlong/*.yaml` (sub-cm facets), in `examples/validation/baseline_crater80N_*.yaml` (those are the
byte-identical regression baselines) or in the `tutorial04_crater80N_*` tutorial configs, and it
must not be carried to Europa or asteroid configs, because $w = 32.5°$ is a lunar regolith number
validated on lunar polar-crater geometry.

The cost is about +22% wall time and 5–6.5 GiB of host RAM on a 140k-face lunar polar mesh
(measured). The memory term scales with the view-factor matrix's nonzero count and reaches an
estimated 22–28 GiB on a 2.4 M-face polar site; measure before enabling it on a large mesh. Every
run with beaming on prints its own `[beam-cost]` breakdown.

Two caveats apply. Beaming makes a run
chunk-decomposition dependent (its emitter/receiver sets are window-scoped; the run
warns), and it does not address the known winter/margin warm biases; those are open and parked.

## flux_cubes

A passive probe that samples the flux field at fixed points in space. The block
is top level rather than nested under `output:`, deliberately, so that it cannot be
reached by the body-defaults machinery that makes `thermophysical:` decorative
on a built-in body. The physics, the geometry cache, the ray budget and
the outputs are all in [flux-cubes.md](flux-cubes.md); this table is the key
list.

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | false | Run the flux-cube pass. Turning it on must be byte-neutral to every mesh field |
| `cache_file` | string | `""` | Geometry cache built by `ptm_flux_cube_geom`. Required when enabled, unless `cache_files` is given |
| `cache_files` | list[string] | — | Multi-height: one cache per cube height, all consumed by one run. Setting this *and* `cache_file` is refused |
| `locations` | list[map] | — | Named locations (`name`, `lat_deg`, `lon_deg`, optional `height_m`, `edge_m`). The parser's only sequence of maps, so its per-entry fields have no dotted keys |
| `write_binary` | bool | true | Write `flux_cubes/flux_cubes.fcb` |
| `write_csv` | bool | true | Write `flux_cubes/csv/cube_XXXXX.csv` |
| `write_csv_max_cubes` | int | 0 | Cap the CSV fan-out on a long window (0 = every cube). Never drops a named location |
| `shadow_probe_rays` | int | 5 | 1 = cube center only; 5 = center plus four rim points |

The geometry keys (`spacing_m`, `height_factor`, `n_rays`, …) are accepted and
reported inert: they live in the cache header, and honoring a YAML copy would
create two sources of truth for one number. `warnUnreadFluxCubeKeys()` names any
key the parser does not consume.

### Cubes at more than one height

A geometry cache carries one height: `FcgHeader.height_m` is a scalar and it
is part of the cache key, so "cubes at 0.35 m, 1.0 m and 2.0 m" is three caches,
not one. Build them with three `ptm_flux_cube_geom --height-m` runs and list them:

```yaml
flux_cubes:
  enabled: true
  cache_files:
    - site_h0p35.fcg
    - site_h1p00.fcg
    - site_h2p00.fcg
```

One thermal solve, three flux-cube passes, three `.fcb` files. The terrain field
does not depend on how high the fixture sits, so re-running the solve per height
would triple the cost of a window to recompute an identical temperature field.

Outputs are named after each cache's own file stem
(`flux_cubes/flux_cubes_site_h0p35.fcb`), and every `.fcb` carries its own
`height_m` in the binary header *and* in the JSON, so each file is
self-describing and `scripts/ptm_flux_cubes_to_hdf5.py` (which takes an explicit
`.fcb` path) needs no change. Two caches whose file-name stems collide are
refused before any pass runs, so no partial product is left behind.

A single cache (whether written as `cache_file:` or as a one-element
`cache_files:`) keeps the historical `flux_cubes/flux_cubes.fcb` name and
produces a byte-identical flux payload either way.

This section is set in the configuration file only; the GUI does not expose it.

## shadow_depth

The shadow-depth pass computes the maximum height above the local surface at which the Sun is not even
partially visible (the *umbral* ceiling), per face and per output epoch. Top
level, for the same reason `flux_cubes` is.

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | false | Run the shadow-depth pass. On or off must be byte-neutral to every other output |
| `h_max_m` | float | 5.0 | Search ceiling, meters. Crew mobility and thermal constraints |
| `march_steps` | int | 16 | March rungs from the surface to `h_max_m` |
| `bisect_rounds` | int | 4 | Bisections inside the bracket the march found. Final bracket = `h_max_m / (march_steps * 2^bisect_rounds)` = 0.0195 m at the defaults |
| `solar_disk_samples` | int | 128 | Disk samples when a unanimously *blocked* probe is escalated |
| `write_frames` | bool | true | Per-epoch `shadow_depth/shadow_depth_<index>.bin` (index zero-padded to at least 6 digits) |
| `monotonicity_faces` | int | 10000 | Faces sampled by the overhang audit (0 disables) |
| `monotonicity_epochs` | int | 10 | Epochs sampled by the overhang audit (0 disables) |

Height is measured along the body radial, not along the face normal. That is the
definition, and it is the same vertical `ptm_flux_cube_geom` places its cubes
along, so the two products stack by construction.

The Sun is treated as a finite disk, and that is what makes this the *umbral* ceiling
rather than the penumbral one. The disk test is the engine's own ladder
(`src/cpp/illumination/solar_disk_visibility.h`, shared with the flux-cube pass).
Dropping the solar angular radius would move the answer by
`d * sec^2(eps) * theta_sun` (0.19 m at a 20 m occluder distance and 45 deg
elevation, ten times the search resolution).

The search is march-then-bisect, not pure bisection. Bisection assumes the
blocked-below / clear-above transition is single-valued, and an overhang breaks
that. The march finds the first clear rung from the surface upward, which is the
contiguous umbral column a person standing there is inside.

Outputs:

| Path | Contents |
|---|---|
| `shadow_depth/shadow_depth_<index>.bin` | one frame per output epoch (index zero-padded to at least 6 digits): `uint32` face count then `float32[F]` |
| `statistics/shadow_depth_{avg,max,min}.bin` | over resolved, nonzero epochs only; `NaN` where a face had none |
| `statistics/shadow_lit_frac.bin` | fraction of evaluated epochs with the Sun visible at the surface |
| `statistics/shadow_unresolved_frac.bin` | fraction umbral above `h_max_m` |
| `statistics/shadow_depth.json` | the manifest: ladder, resolution, sentinel encoding, solar-disk treatment, the resolved ray `tnear`, the epoch window, and the consistency-gate counts. Written last, so its presence certifies the maps beside it are complete |

Sentinels: `0.0` means the Sun is visible at the surface; `+inf` (float32
`0x7F800000`) means no clear height was found at or below `h_max_m`, which is
the majority value inside a permanently shadowed region by design (no number
is invented for it; the fractions report it structurally); `NaN` means the face
was not evaluated.

In the GUI's 3D viewer, these six products (the per-epoch series, the three
statistics and the two fractions) appear in the Field drop-down like any other.
Because two of the three encodings are not on the color ramp, the shadow-depth
fields are scaled to `[0, h_max_m]` rather than autoscaled (measured through
the real lookup table, an autoscale over `+inf` maps every finite value *and*
`+inf` to one color), and the sentinels are drawn off-scale: gray for `NaN`
(all six fields), and for `+inf` a color chosen against the colormap in use,
magenta by default. No fixed color is safe: magenta is exactly the top of
`cool` and near a point on `hsv`, both of which the Colormap dropdown offers, so
the viewer picks the first candidate the ramp does not contain and says which
one it picked. A pane above the view carries that legend for the selected field,
and warns if no candidate is distinguishable. The array itself is never
rewritten, so a probed facet still reports `inf`.

For chunked runs, `scripts/run_chunked_simulation.py` force-sets
`output.skip_time_series` for every chunk, so it refuses a config that asks
for `write_frames: true`. Set `write_frames: false` (the window statistics and the
manifest still merge correctly across chunks, from `statistics/_raw` sum/count
side files) or run the window unchunked.

## gpu

Cross-cutting GPU safety policy: whether a GPU rung is allowed at all, and
which devices each pool may use. This is distinct from the per-solver
`gpu_devices` / `gpu_device_ids` / `gpu_batch_size` keys, which tune a rung that
has already been allowed. The full rationale, the platform-gating table and the
cost of each guardrail are in
[gpu-wedge-guardrails.md](dev/gpu-wedge-guardrails.md).

| Key | Type | Default | Description |
|---|---|---|---|
| `min_vram_margin_gib` | float | 2.0 | Floor on the per-device VRAM headroom a resident radiosity CSR replica may leave behind. A rung with less is refused (see gating below). 0 = fit check only, the pre-2026-08-22 behavior |
| `placement_policy` | string | `auto` | `auto`, `shared` or `disjoint`: whether the radiosity replica and the GPU thermal pool may occupy the same devices |
| `placement_margin_gib` | float | 4.0 | Margin below which `auto` separates the two pools onto disjoint devices, and below which the thin-margin advisory prints |
| `thin_margin_policy` | string | `separate` | What a margin below `min_vram_margin_gib` does: `separate` the two pools onto disjoint devices and keep the GPU rung, or `refuse` the rung outright |
| `hard_min_vram_margin_gib` | float | 0.25 | Under `separate`, the floor below which the GPU rung is refused anyway |

The failure these keys guard against (a wedged GPU that takes
the machine down with it) was measured on macOS with AMD (Navi21) devices, so the policy is gated by platform.

| | macOS + AMD | every other host |
|---|---|---|
| `min_vram_margin_gib` | enforces: a thin rung is refused and radiosity runs on the CPU | advisory: prints the same message and proceeds |
| `placement_policy: auto` | separates the pools when the margin is thin | never separates; prints a note when it would have |
| `placement_policy: disjoint` | separates | separates (an explicit instruction is obeyed everywhere) |

Setting `min_vram_margin_gib` in the config (or `PTM_GPU_MIN_VRAM_MARGIN_GIB`
in the environment) makes it enforcing on every host: an operator who asks
for a policy gets the policy.

With the 2.0 default, a site whose CSR replica fits by less than 2 GiB
is treated as thin, and by default (`thin_margin_policy: separate`) the engine
moves the radiosity replica and the GPU thermal pool onto disjoint devices
and keeps the GPU rung, at a derived +7-10 min per window on a multi-million-face polar site. It gives up
the rung only below `hard_min_vram_margin_gib` (0.25), or on a rung that cannot
be separated. That matters because the alternative is expensive: measured on
lunar polar site B (which fit by 0.64 GiB), CPU radiosity cost 108.9 min/window against
31.4 min on the GPU pool, 8 windows each. The reason the margin needs guarding
at all is that the GPU thermal pool allocates roughly 0.77 GiB per device on
the *same* dies while the replica is resident, and the fit check does not count
it. Set `thin_margin_policy: refuse` to give up the rung instead; see
[gpu-wedge-guardrails.md](dev/gpu-wedge-guardrails.md).

`auto` engages only when all of the following hold: the thermal solver is on the GPU, the two
device sets overlap, the margin is below `placement_margin_gib`, the host is
macOS + AMD, and neither side pinned explicit device ids. An explicit pin is an
operator decision and `auto` will not overrule it; it warns instead. The
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
| `map_depths_m` | list of float | `[]` | Depths (m) at which to write mesh-wide temperature maps. `[]` = off |
| `map_depth_snap_to_node` | bool | true | Snap each requested depth to the nearest solved grid node |
| `map_depth_stats` | list of string | `[avg, max, min]` | Which statistics to write per depth |
| `skip_mesh` | bool | false | Do not copy `mesh.ply` into the output directory |

### Two-band radiosity and effective-albedo products

Three products are written without a key of their own, by whichever run
produces them:

| On disk | Written by | Shape |
|---|---|---|
| `indirect_solar/`, `statistics/indirect_solar_{avg,max,min}.bin` | a two-band radiosity run | per-step series + statistics |
| `indirect_ir/`, `statistics/indirect_ir_{avg,max,min}.bin` | a two-band radiosity run | per-step series + statistics |
| `statistics/effective_albedo.bin` | every run with a per-face flux path | one value per face |

All three appear in the GUI's 3D viewer Field drop-down. Two things about them
are not visible in the numbers:

- The indirect bands are absorbed flux at the facet, while
  `direct_solar`, `primary_reflected` and `primary_ir` (same units, same mesh,
  next to them in the same menu) are stored incident. That is the engine's
  own convention: the two-band solve produces exactly these absorbed band
  fluxes, and `indirect_solar + indirect_ir` equals the value coupled into the
  thermal boundary condition (the engine measures that identity on every run).
  The viewer says so in the pane above the view when one of them is selected.
- `effective_albedo` is `NaN` wherever a facet was never lit during the
  integration window; there is no incident flux to form the ratio from, so the
  engine writes no number rather than inventing one. On a 2,397,890-facet lunar
  polar run at 1 h cadence that is 26,057 facets, exactly the count the run's
  own log reports as never lit. The viewer scales the color bar over the finite values only and
  draws those facets gray, with a legend saying what gray means, so the map
  does not read as broken.

### Subsurface temperature maps (`map_depths_m`)

Mesh-wide temperature statistics at depth, written beside the surface ones
and reduced over the same integration window:

```
statistics/temperature_z0p050m_{avg,max,min}.bin   uint32 face count + float32 per face
statistics/depth_map_z.json                        what was asked for, what was sampled
statistics/_raw/temperature_z0p050m_{sum,count}.bin  (with write_raw_stats, for the chunk merge)
```

The file name carries the resolved depth to three decimals with `.` written
`p`, so it is sortable, filesystem-safe and reversible. Two requests that
resolve to the same name are refused rather than silently overwriting.

`[]` is off and is byte-neutral. Turning the product on changes no other
output file; that is an acceptance test of the feature, not a side effect.

Depth maps do not appear in the GUI's 3D viewer Field drop-down. They
export (GeoTIFF/HDF5/COG) and read from Python normally, and
`ptm.results.detect_available_statistics` reports them, but the viewer's field
menu is built from a static catalog and these names carry the depth, so no
static catalog can hold them.

Both list keys must be written as YAML lists, and the engine refuses anything
else. `map_depths_m: 0.05` and `map_depths_m: "0.05, 0.25"` are scalars, and a
scalar iterates as an empty sequence. Before 2026-09-06 that produced a
complete run, exit 0, with no depth maps and nothing in the log; the scalar form
of `map_depth_stats` reached a refusal that reported the list as *empty* when the
user had in fact named a statistic. Both now stop the run at config load with a
message naming the fix. `map_depths_m:` with nothing after it stays legal and
means the same as `[]` (off). Nothing is coerced: a quoted `"a, b"` is one value,
not two. The same rule now covers every list key; see
[Keys the engine reads as lists](#keys-the-engine-reads-as-lists).

A depth below the deepest solved node is refused naming `thermal.thermal_b`,
which is the YAML key. The engine's own field for it is `grid_b`; the two are
different strings and only the first is something you can put in a config file.

Snapping to the nearest node is the default. A depth between two grid nodes is
sampled by linear interpolation, and the interpolation error is not small: at a
lunar-equatorial diurnal amplitude *A* = 160 K it is ~0.195 K for an
instantaneous value, which is over this project's 0.15 K reference floor by
itself, and it grows with harmonic number because harmonic *h* propagates with
skin depth `delta/sqrt(h)`. Only a depth landing on a node avoids it, so the
default snaps and reports the snapped depth plus the residual. Set
`map_depth_snap_to_node: false` to sample the depth as asked; the run then
prints what that costs, and `depth_map_z.json` carries the per-harmonic budget.

avg, max and min do not have the same error budget. Averaging commutes with
interpolation in *z* exactly; max and min do not, because adjacent layers peak at
different times. That is why `max`/`min` are accumulated in the solver at the
requested depth rather than interpolated from per-layer extrema afterwards;
the latter costs `1.229e-3 * A` and is worst at layer 0. `depth_map_z.json`
therefore reports two separate coefficient arrays per depth, and they are not
interchangeable: `bias_instantaneous_per_A_by_harmonic` is a rigorous upper bound
(on a time series and, since `|max f - max g| <= max|f - g|`, on the maps too),
while `bias_max_per_A_by_harmonic` is the smaller typical error in the max.

z = 0 is the surface product. The `z=0` map is the same quantity as
`statistics/temperature_{avg,max,min}.bin`, sampled at the same instants from the
same re-evaluated surface node. On the CPU SIMD-batch path all three are
byte-identical to their surface counterparts; on the GPU path `max` and `min` are
byte-identical and `avg` differs by the device's float32 running-sum error
(measured 7.6e-5 K over a 118-frame window). Use that identity as a plumbing
check on any new run.

Several solver paths carry it: the CPU SIMD-batch solver (the shipped default),
the GPU persistent solver on any device count, the GPU radiation-coupling
selected-faces re-thermal, and, since 2026-09-07, both CPU scalar paths
(`--no-simd-batch`, and the streamed fallback it drops into on a large mesh),
*provided no per-face thermal property map is loaded*. A radiation-coupled run on
the CPU solver still refuses. Every refusal names the fix.

The CPU scalar path has a condition because it builds a `Profile`, and therefore
a depth grid, per face. The only per-face input to that grid is `H`, so without a
per-face `H` map every face lands on the same grid, the plan resolved against the
reference grid describes all of them, and the product is the SIMD-batch product
computed one face at a time. That is not assumed: the run compares each face's
own grid against the plan's before folding that face in, and stops if any face
disagrees. With `thermal.h_parameter_map`, `thermal.thermal_inertia_map` or
`thermal.use_default_property_map` set, the grids genuinely differ and the run
refuses, before the illumination phase where the configuration alone settles it
and at the solver otherwise.

That refusal is a decision about the *product*, not only about the file name.
Snapping each face to its own nearest node would leave the name (one depth)
describing a surface that undulates facet to facet. Turning snapping off repairs
the name (every facet is then interpolated to exactly the requested depth) but
not `depth_map_z.json`, which carries one `native_z_m`, one `node_j`, one
`weight_on_j_plus_1` and one interpolation-error figure per depth, each of which
would be a distribution over facets. Publishing one facet's numbers as the
manifest for a map assembled from many is the class of defect the rest of these
refusals exist to prevent. A map on a per-face grid needs a per-face error budget,
and this engine does not write one.

Measurements on the shipped 12,800-facet lunar crater at depths 0, 0.05 and
0.25 m show the following. The z = 0 map is byte-identical to `statistics/temperature_*.bin` on the
CPU scalar path, as it is on the others; there the surface node it publishes *is*
the column's node 0, so no re-evaluated surface value is substituted. Against the
CPU SIMD-batch path at the same depths, the two solvers' maps agree to within the
margin by which their surface products already differ: surface `avg`
max |ΔT| 0.205 K, and at 0.053 m and 0.218 m 0.137 K and 0.118 K. The depth
sampling adds nothing beyond the solvers' own disagreement.

For chunked runs with `write_raw_stats: true`, `ptm_merge_chunks` discovers the
depth fields from chunk 0 and refuses if any chunk carries a different depth set
or a different statistic set. `depth_map_z.json` is carried through to the merged
directory.

Two products are not implemented here, and are deliberately absent rather than present-and-dead:
a per-depth time series (5 depths x 1440 frames at 2.4M faces is 64 GiB, a
disk event rather than a product) and an arch-B native-grid export.

### Per-location depth series (`locations_file` + `depths_file`)

Setting both keys writes one CSV per named site,
`locations/<name>_depths.csv`: a `# Location:` comment, a `# Integration:`
comment, a header `time,T(z=0.000 m),T(z=0.050 m),...`, then one row per step
of the whole integration, equilibration prefix included (unlike
`statistics/`, which covers the output window only).

> ### Announced behavior change, 2026-09-07 (read before comparing runs)
>
> Two things changed for every run that writes a `*_depths.csv`. Both are
> deliberate, both move numbers, and neither is a bug fix.
>
> **1. The CSV is now the mesh solver's own column, not a second integration.**
> Before this release the depth CSV at a site was solved by a *separate* 1-D
> thermal run of that site, with its own equilibration seed window, its own surface
> anchor and its own absorbed-flux composition. `statistics/temperature_*.bin` at
> the same mesh face came from the whole-mesh solver. The two need not agree, and
> did not. The CSV is now recorded *during* the whole-mesh solve, so the file and
> the map are one integration.
>
> *How much this moves the numbers* (global lunar icosphere, 4 sites, 6 depths,
> one equilibration lunation, CPU SIMD-batch, the shipped default solver):
>
> | | max \|ΔT\| | where |
> |---|---|---|
> | all rows, including the equilibration prefix | 8.71 K | z = 0.010 m |
> | output window only | 1.40 K | z = 0 m |
> | output window, mean over rows | 0.005 – 0.25 K | deepest to shallowest |
>
> On a radiation-coupled lunar crater run the same comparison is 0.81 K / 0.70 K.
> On the CPU scalar path (`--no-simd-batch`) the CSVs are unchanged to the last
> printed digit; that path already used the same column routine and the same
> flux, so there was nothing to reconcile.
>
> The new values are the ones the rest of the run reports: after the change the
> CSV, reduced over the statistics window, matches
> `statistics/temperature_*.bin` at the same face to within the two files'
> combined representation floor (float32 in the `.bin`, six decimals in the CSV),
> measured at 0.945x that floor, against 7,400x before.
>
> **2. A requested depth below the deepest solved node is now refused.** The CSV
> used to extrapolate below the grid with Fourier's law,
> `T = T_bottom + Qb*(z - z_bottom)/k_bottom`, and a warning. It no longer does:
> the run stops, before the solve, naming the deepest depth it can report and
> `thermal.thermal_b` (the YAML key; the C++ field is `grid_b` and is not
> settable). This is the depth map's D9 rule, extended to every depth product.
>
> This will stop configs that used to complete. The shipped
> `data/locations/depths.txt` asks for depths down to 1.00 m, and the default
> lunar grid (`thermal_b: 15`) solves to 0.4706 m; that pairing now refuses.
> The fix is in the message: raise `thermal.thermal_b` until the bottom node
> reaches the depth you want (on the Moon, `45` reaches 1.4119 m), or remove the
> out-of-range depths. The refusal costs seconds, not a whole run.

Which integration produced a given file is written in the file. Line 2 of
every CSV is either

```
# Integration: mesh column, recorded during the whole-mesh solve on the CPU SIMD batch path (grid fingerprint ...); the same integration as statistics/temperature_*.bin at this face
```

or

```
# Integration: standalone 1-D column, re-solved for this site (runThermalModelDepthSeries) -- NOT the mesh-wide integration; see the run log
```

A log line does not travel with a CSV, and after this change the same file name
can hold either product. Every reader in this repository skips `#` lines, so the
extra comment is additive.

Three solver paths record the column: the CPU SIMD-batch solver (the shipped
default, including its streamed variant), the CPU scalar solver, and the CPU
radiation-coupling selected-faces re-thermal. A coupled CPU run's CSV therefore
carries the *re-thermalled* column at the faces coupling touched, not iteration
zero's. The GPU persistent solver keeps no per-frame column and is not
covered: a GPU run falls back to the standalone integration, says so in the log,
and stamps the fallback into each CSV. It is not silent and it is not a refusal:
the file is still written, and still usable, as long as you read it as what it
says it is.

The other case that falls back is the CPU scalar path with a per-face
thermal-inertia / `h_parameter_map`, which gives every face its own depth grid.
The recorder compares each face's node depths against the reference grid
*elementwise*, not by layer count (two faces can agree on the number of layers
and differ in the top cell), and stands down naming which of the two checks
refused.

The CSV composes with `map_depths_m` as follows. The two products share the solved
column, and where both are active they are the same numbers: measured over
4 sites x 4 depths x {avg, max, min}, the CSV reduced over the statistics window
matches the depth map at that site's face to 0.925x the two files'
representation floor on the CPU scalar path and 0.945x on the CPU SIMD-batch
path (0.962x with a per-face `H` map on the SIMD path). They reach that
number through two independent samplings (the map through
`depthmap::Accumulator::addColumn`, the CSV through `depthmap::sampleColumnAtDepths`),
so the agreement is a cross-check, not a tautology.

Where they *refuse*, the stricter one wins and it is the map. On the CPU scalar
path with a per-face `H` map, `map_depths_m` refuses the whole run at
configuration time; the depth CSV's own stand-down is never reached, because
there is no run left to stand down in. Drop `map_depths_m` and the same config
completes, with the CSV falling back to the standalone integration and saying
so. Both behaviors are correct and they are consistent: neither product is ever
written from a grid it was not resolved against.

A per-face H map changes what the CSV means, deliberately. The recorded
column is the mesh solve's, and the CPU SIMD-batch solver puts every face on one
grid at the map's reference *H*. The old standalone column was solved at *that
face's* H on its own grid: better resolved for the site, and not the number the
mesh-wide field reports there. Agreement with the mesh is what the CSV is for;
the run prints both H values at each site so the trade is visible.

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
| `H_override` | float | -1.0 | Regolith density/conductivity e-folding depth H, in meters: the scale in `rho(z) = rhod + (rhos-rhod)·exp(-z/H)` and `kc(z) = kd + (ks-kd)·exp(-z/H)`. Inert when `constant_thermal_inertia > 0` (which makes those exponentials multiply zero) and out-ranked per face by `thermal.h_parameter_map` / `thermal.thermal_inertia_map`; both cases warn |
| `incidence_angle_albedo` | bool | false | Enable the Hapke directional-hemispheric albedo |
| `albedo_model` | string | `""` | `hapke` is equivalent to the key above |
| `albedo_hapke_w` | float | -1.0 | Single-scattering albedo (below 0 = derive from the Bond albedo) |
| `conductivity_model` | string | `"hayne2017"` | `hayne2017` or `martinez2021`; see [martinez-conductivity-model.md](martinez-conductivity-model.md). An unrecognized value stops the run |

## Keys the engine reads as lists

Seven keys are read as YAML sequences. yaml-cpp iterates a scalar as an *empty*
sequence, so before 2026-09-06 writing one without brackets was a silent no-op:
the value was dropped, the default stood, and nothing said so. All seven now stop
the run at config load, naming the value given and the list form that fixes it.

| Key | A scalar used to mean | Now |
|---|---|---|
| `output.map_depths_m` | product off; complete run, exit 0, no maps, no message | refused |
| `output.map_depth_stats` | empty; run stopped later reporting the list as *empty* | refused, as a shape error |
| `thermal.gpu_device_ids` | no explicit pin | refused |
| `illumination.gpu_device_ids` | no explicit pin | refused |
| `radiation.gpu_device_ids` | no explicit pin | refused |
| `generic_body.shape.semi_axes_km` | body keeps its default shape | refused |
| `flux_cubes.cache_files` | no cache loaded; run produces no flux cubes at all and exits 0 | refused |

The `gpu_device_ids` case is a safety issue, not a tidiness one.
`gpu_device_ids` overrides `gpu_devices` *only while it is non-empty*. A scalar
therefore does not pin a different card; it drops the pin, and the run follows
`gpu_devices`: `1` (the default) selects the first enumerated GPU, and `0`/auto
selects every enumerated GPU. So `gpu_devices: 0` beside `gpu_device_ids: 2`
silently meant "use all of them".

`semi_axes_km` is refused on length as well: it must list exactly three axes.
A two- or four-element list was also skipped in silence, leaving the body at its
default shape while the file appeared to set it.

A key written with nothing after it (`gpu_device_ids:`, `map_depths_m:`) is a
null, not a scalar, and keeps meaning "not set"; that spelling is unchanged.
Nothing is ever coerced: a quoted `"0, 1"` is one value, not two.

## generic_body

Bodies without built-in properties are described here, with an ephemeris from
either Keplerian elements or a SPICE metakernel, plus a `surface:` block that
carries the full thermophysical property set. See the asteroid examples in
[`../examples/`](../examples/) and
[docs/tutorials/](tutorials/).

| Key | Type | Default | Description |
|---|---|---|---|
| `shape.semi_axes_km` | list[float] × 3 | body default | Triaxial semi-axes a, b, c in km. Must be exactly three entries; any other shape or length is refused (see [Keys the engine reads as lists](#keys-the-engine-reads-as-lists)) |
| `shape.type` | string | spherical | `spherical`, `ellipsoidal` or `custom_mesh`; any other value is refused. `ellipsoidal` without three positive axes, or `custom_mesh` without a mesh, builds a sphere and says so |
| `ephemeris_source` | string | auto | `keplerian`, `spice` or `auto`; any other value is refused. `auto` resolves to Keplerian (a warning is printed if a `spice_metakernel` is also set) |
| `orbit.epoch` | epoch string | J2000 | Epoch of the osculating elements; `orbit.mean_anomaly_deg` is the mean anomaly at this epoch. See the epoch formats below |
| `spin.prime_meridian_deg` | float | 0.0 | Prime meridian angle W0 at `spin.epoch`, degrees about the pole, measured from the body's reference axis: the J2000 ecliptic +x (the vernal equinox) projected onto the body's equatorial plane, or ecliptic +y if the pole lies within 25.8° of +x (`orbital/body_fixed_frame.cpp`). W(t) = W0 + 360° (t − epoch) / `spin.period_hours` |
| `spin.epoch` | epoch string | J2000 | Epoch of `spin.prime_meridian_deg` |

Before 2026-09-24, `orbit.epoch` and `spin.prime_meridian_deg` were parsed and
ignored: every generic body rotated from W = 0 at J2000 and applied its mean
anomaly at J2000. A config that set them to anything but the
J2000 defaults produces a different, now correct, run.

Epoch formats (both `*.epoch` keys, all on the TDB scale, the scale of
SPICE ET and of JPL osculating-element epochs): `YYYY-MM-DDTHH:MM:SS[.fff]`
(a space may replace the `T`; the time and seconds are optional), the same with
a ` TDB` suffix, or a Julian date `JD 2451545.0` (the `JD` may be omitted for a
number above 100000). A `UTC`, `Z`, `TT`, `TAI` or `UT` suffix is refused rather
than converted. So `'2000-01-01T12:00:00'`, the value every shipped config
carries, is J2000 exactly.

A DAMIT spin state (`lambda beta P` / `JD0 phi0`) maps onto these keys by
`spin.epoch: JD <JD0>`, the pole from (lambda, beta) converted to equatorial
RA/Dec, and `prime_meridian_deg = phi0 + delta`. DAMIT's zero of phi and this
engine's zero of W are different directions in the body's equatorial plane,
so delta, the angle about the pole between them, is a constant for a given
pole; compute it once from the two reference vectors, and confirm the result
(sign included) by comparing the sub-solar point at one epoch against an
independent prediction.

Any key under `generic_body:` (or its `shape:`, `orbit:`, `spin:`, `surface:`
blocks) that the engine does not read is named in a `[config] WARNING` at load
time. The HORIZONS writer's provenance keys (`spk_file`, `surface.albedo_*`
other than `albedo`) are named in a `NOTE` instead.

### How `generic_body.spice_metakernel` is resolved

| Value | Resolved to |
|---|---|
| Absolute path (`/…/metakernel.tm`) | Used exactly as given; the kernel directory is not prepended |
| Relative `data/…` path (`data/horizons_cache/…/metakernel.tm`) | The same `PTM_DATA_ROOT` contract as `mesh.import_file` and every other data key: `$PTM_DATA_ROOT`, else the repository `data/` found from the config file's directory |
| Any other relative path (`bennu.tm`) | `<kernel dir>/<value>` first, then `<current working directory>/<value>` |

The kernel directory is itself resolved as: `$PTM_SPICE_KERNELS`, then
`$PTM_DATA_ROOT/kernels`, then the compiled-in `SPICE_KERNELS_DIR` (CMake cache
variable, default `<repo>/data/kernels`); see
[ENVIRONMENT.md](ENVIRONMENT.md). There is no config key or command-line flag
for it in `ptm_thermal_sim` (`ptm_camera_sim` has `--kernels-path`); each rung
replaces the compiled-in directory rather than adding to it, so an override
directory must contain the metakernels as well as `lsk/`, `pck/` and `spk/`.

`$PTM_SPICE_KERNELS` is taken exactly as given, populated or not, because it
is an explicit instruction and demoting it would hide a typo. The two implicit
rungs are different: a directory with no `lsk/naif0012.tls` cannot serve any run
(not even a Keplerian one, since `start_time` still has to be converted), so it is
passed over in favor of the next rung. That is what lets a fresh clone or
worktree, which ships the `*.tm` files but not `lsk/`, `pck/` or `spk/`, find a
kernel tree through `$PTM_DATA_ROOT` instead of dying several screens later in
`str2et_c` with a bare `SPICE(NOLEAPSECONDS)`.

Every run prints the kernel directory it resolved and its source (`SPICE kernel
directory: … ($PTM_DATA_ROOT/kernels)`) and the metakernel file it actually
loaded (`Loaded SPICE meta-kernel: …`). If none of the candidates exists the run
stops and lists every path it tried (both the metakernel candidates and the
whole kernel-directory ladder, in order, with the reason each rung was or was
not taken). It never substitutes a different metakernel for the one you
configured.

(Separately, and not covered by that guarantee: a `simulation.body` name other
than `Moon`, `Mercury` or `Europa` that is *not* backed by a `generic_body:`
block is refused at startup, with `rc 1` and a message naming the three
implemented bodies and the generic-body route. Until 2026-08-26 such a name fell
through to Europa's property set and Europa's metakernel instead, and the run
completed with `rc 0` and zero absorbed flux on every face. Use `generic_body:`
for anything outside the three built-ins.)

Paths inside the metakernel (`PATH_VALUES`) are resolved by CSPICE itself,
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
simulated, no output directory is created, and an existing output directory
(including its `heartbeat.txt`) is left alone.

It covers YAML syntax and structure, the `PTM_DATA_ROOT` resolution contract,
the enum values (`conductivity_model`, `coupling_mode`, `exchange_selection`,
`gpu.placement_policy`, `gpu.thin_margin_policy`), the numeric ranges
(`nearfield_quadrature_order`, `min_shadow_fraction`, the `gpu.*` margins), the
`beaming`/`rms_slope_deg` pair, and whether this build has the GPU support the
config asks for.

It does not cover anything that needs the mesh, the body or SPICE: mesh
loading and units, generic-body construction, SPICE kernels, property-map face
counts, the `martinez2021` density guard, view factors, and the spin-up seed
strings. The mode prints that boundary every time it runs. `rc 0` means the
configuration is well formed, not that the run will complete.

The *Check Configuration* item in the GUI's *Simulation* menu uses this and shows the engine's
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
is refused at startup unless the config has a `generic_body:` block (see the
SPICE section above). With that block, the run seeds placeholder properties,
says so on stderr, and overwrites them from the block.
