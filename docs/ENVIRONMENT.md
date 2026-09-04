# PTM3D environment variables

Every `PTM_*` environment variable read by the C++ binaries and the shipped
scripts. Almost all are **unset by default**, and for those, unset reproduces the
established baseline behavior (numerically identical to the last digit wherever a
reference baseline exists). The exceptions are the **[kill-switch]** entries
below: those name a behavior that is now ON by default, and it is `=0` — not
unset — that restores the older behavior. To re-audit this table against the
tree, grep `src/` and `scripts/` for `getenv("PTM_` and compare the result
against the names below.

Status tags:

- **[recommended-ON]** — validated fixes that default OFF only so historical
  baseline runs can still be reproduced digit-for-digit. **The validated set is:**
  `PTM_ILLUM_TRUE_NORMALS=1 PTM_ILLUM_RISE_EPS=1 PTM_ILLUM_RISE_KEPS=0.2
  PTM_ILLUM_RISE_HABS=0 PTM_ILLUM_SEAL_HOLES=1 PTM_EQ_PERFACE_FLUX=1`.
  New science runs should set these.
- **[kill-switch]** — the behavior is ON by default; the variable exists so `=0`
  can restore the older behavior for a regression comparison. Unset is **not**
  the old baseline for these. They accept `1`/`true`/`on`/`yes` and
  `0`/`false`/`off`/`no` (case-insensitive); **any other value is fatal at
  startup**, before the illumination pass, rather than guessed at — an explicit
  `=1` can make an unsupported path refuse to run, so `=false` must not be read
  as a request for it.
- **[standard]** — a stable, supported knob; set when the situation calls for it.
- **[tuning]** — performance/memory lever; defaults are sensible.
- **[experimental]** — implemented and gated, but not yet part of the validated
  production recipe; understand the linked notes before using.
- **[internal]** — diagnostics and campaign-runner plumbing; not for end users.

## Setting them from the config: the `engine_env:` block

A config may declare the variables its run needs, and **the engine applies them
itself** at startup (since 2026-08-04; before that the block was documentation
only, and a regression pair could silently run under two different environments):

```yaml
engine_env:
  PTM_ILLUM_TRUE_NORMALS: '1'
  PTM_ILLUM_RISE_EPS: '1'
  PTM_ILLUM_RISE_HABS: '0'
  PTM_ILLUM_SEAL_HOLES: '1'
```

Rules:

- The block is applied **before** the configuration itself is parsed, so it can
  even set `PTM_DATA_ROOT`.
- **The process environment wins.** A variable already exported to a different
  value is left alone and the disagreement is printed as `CONFLICT`, so a
  deliberate command-line override still works.
- Every variable is echoed with its disposition (`applied`, `already set`,
  `CONFLICT`), so the run log records the environment the physics actually saw.
- The block is only read by `ptm_thermal_sim`. Other binaries and the analysis
  scripts still need the variables exported.

## Paths and data

| Variable | Default | Effect | Status |
|---|---|---|---|
| `PTM_DATA_ROOT` | unset | Points at the science-data directory when it lives outside the repository. Every relative `data/...` path in a config resolves in this order: an absolute config path always wins → `$PTM_DATA_ROOT` → the repository's `data/` (found by walking up from the config file to the directory containing `data/data_manifest.json`). If a required input is still missing, the program stops with an error naming the dataset and the exact `scripts/fetch_data.py` command. | [standard] |
| `PTM_SPICE_KERNELS` | unset | Overrides the SPICE kernel directory when none was set by config/CLI. Resolution ladder: explicit setting → this env → `$PTM_DATA_ROOT/kernels` → compiled-in `SPICE_KERNELS_DIR` (default `<repo>/data/kernels`). The first two rungs are explicit instructions and are taken exactly as given; the last two are skipped if they hold no `lsk/naif0012.tls`, so a fresh clone or worktree (which ships the `*.tm` files but not `lsk/`) can still reach a kernel tree through `$PTM_DATA_ROOT`. Every run prints the directory it chose and its source, and any SPICE failure lists the whole ladder in order. | [standard] |
| `PTM_ALBEDO_MAPS_DIR` | unset | Overrides the albedo/property FITS map directory (else the config's `albedo_maps_directory`). | [standard] |
| `PTM_MACHINE_PROFILE` | `~/.ptm3d/machine_profile.json` | Path to the per-host GPU machine profile written by `--calibrate-gpu`. | [standard] |
| `TMPDIR` | `/tmp` | (POSIX, not `PTM_*`.) Honored for the H-matrix out-of-core scratch files. | [standard] |

## Illumination (shadow-ray correctness fixes)

| Variable | Default | Effect | Status |
|---|---|---|---|
| `PTM_ILLUM_TRUE_NORMALS` | off | Use the mesh's true triangle-winding normals for illumination instead of the radial-up flip. The flip inverted ~97% of steep-wall faces on polar terrain, fabricating sunlit overhangs. Validated: removes the false >380 K overhang tail without darkening any genuinely sunlit face. Ignored (with a printed message) on a smooth global icosphere, whose normals are analytic radial ones substituted deliberately to remove a pentagon-vertex bias and which has no overhang faces to correct. The startup line reports how many normals were actually overridden, or that the override was declined. | **[recommended-ON]** `=1` |
| `PTM_ILLUM_RISE_EPS` | off | Enables the out-of-plane-rise self-relief discriminator for shadow rays: separates a face's own mm-scale relief from a real occluding rim by rise height rather than the legacy 1 cm ray offset (which hopped real rims and leaked direct sun into PSRs). | **[recommended-ON]** `=1` |
| `PTM_ILLUM_RISE_KEPS` | 0.2 | Facet-edge fraction defining the edge-proportional self-relief floor for RISE_EPS. | **[recommended]** leave at `0.2` |
| `PTM_ILLUM_RISE_HABS` | unset (edge-proportional) | Absolute self-relief floor override in meters; `0` = exact source-triangle-only exclusion. Validated: `=0` eliminates 100% of the grazing-incidence light leaks without introducing spurious self-shadowing speckle. | **[recommended-ON]** `=0` |
| `PTM_ILLUM_SEAL_HOLES` | off | Fills interior mesh holes with shadow-only faces (never solved/output) so finite-disk penumbra rays cannot thread through mesh gaps (the "crater-center flash"). Validated on a 509k-face lunar polar mesh: 49 shadow-only fills, flash eliminated, one collateral face in 778. | **[recommended-ON]** `=1` |
| `PTM_ILLUM_RISE_EPS2` | off | "option 3" multi-hit refinement sub-mode (implies RISE_EPS). Superseded by the RISE_EPS + RISE_HABS=0 recipe; did not fix the spurious-illumination classes on its own. | [experimental] |
| `PTM_ILLUM_RISE_BARY_EPS` | internal default | Barycentric tolerance for the RISE_EPS source-triangle exclusion. | [tuning] |
| `PTM_ILLUM_TOPO_SELFSKIP` | off | Alternative mitigation: topological self-skip of the source face's neighborhood (CPU illumination path only; does not engage on the GPU tracer). | [experimental] |
| `PTM_ILLUM_SELFSKIP_RINGS` | 1 | Neighborhood ring count for TOPO_SELFSKIP. | [experimental] |
| `PTM_ILLUM_HORIZON_CULL` | off | Heightfield-horizon pre-cull of shadow rays. | [experimental] |
| `PTM_ILLUM_HORIZON_SLACK_DEG` | internal default | Slack angle for HORIZON_CULL. | [experimental] |
| `PTM_ILLUM_HORIZON_CELL_SCALE` | internal default | Heightfield cell scale for HORIZON_CULL. | [experimental] |
| `PTM_DISABLE_STRICT_HORIZON` | 0 | `=1` disables strict outward-normal enforcement when computing face geometry. Debug escape hatch. | [internal] |

## Thermal equilibration

| Variable | Default | Effect | Status |
|---|---|---|---|
| `PTM_EQ_PERFACE_FLUX` | off | Seeds each column's equilibration from **its own** time-mean absorbed flux + geothermal Qb instead of the legacy first-10-step instantaneous flux with a 100 K nightside floor. PSR floors cold-start at the geothermal steady state (~24.7 K on the Moon) instead of being stamped warm they cannot cool down from within a run. Unset, results are numerically identical to the last digit of the historical baseline; the bowl-crater validation test passes either way. | **[recommended-ON]** `=1` |
| `PTM_EQ_WITH_INDIRECT` | off | Folds the (phase-mapped) indirect radiation flux into re-equilibration for coupled seasonal runs. Distinct from `thermal.eq_indirect_source`, which corrects the initial whole-mesh equilibration once; this one re-equilibrates the selected faces inside every coupling iteration. They compose (each acts on a different phase) but the combination is not characterized. | [experimental] |
| `PTM_EQ_INDIRECT_CYCLES` | 2 | Number of prior seasonal cycles assumed stationary for the EQ_WITH_INDIRECT phase mapping. | [experimental] |
| `PTM_EQ_RETHERMAL_BATCH` | 4000 | GPU batch cap during EQ_WITH_INDIRECT re-equilibration (the 50k default overruns the macOS/AMD GPU command-buffer watchdog). | [tuning] |
| `PTM_EQUILIBRATION_CYCLES` | unset (config wins) | Number of passes over the equilibration forcing array, GPU persistent solver only. Overrides `thermal.equilibration_cycles`; the solver's own value is 2. Must be an integer ≥ 1 — anything else is reported as an error rather than silently ignored. | [standard] |
| `PTM_EQ_LEGACY_SEED` | unset | Read by nothing in this tree. Setting it changes no result; the CPU batch path prints a warning naming it so a caller is not left assuming it took effect. | [internal] |
| `PTM_RETHERMAL_WARMUP_SKIP` | 0 | Number of leading output steps of a coupling re-run that keep the un-coupled iteration-0 temperature. `=2` restores the pre-v1.0.3 mask, which existed to hide a checkpoint-seed transient that no longer occurs. | [internal] |

## Radiation exchange / radiosity

| Variable | Default | Effect | Status |
|---|---|---|---|
| `PTM_RAD_TWO_BAND` | unset | Alternative activation path for band-separated radiosity, which is **already the default** through `radiation.band_separated: true`. The visible channel scatters with the Bond albedo, the IR channel with 1−ε, and the absorbed indirect flux is their sum. **Requires `PTM_SOLAR_ALBEDO`**: without a per-face Bond albedo the request warns and stays gray. Set `radiation.band_separated: false` to get the single gray band. | [standard] |
| `PTM_LEGACY_RADIOSITY_STATIC_ALBEDO` | 0 | `=1` builds the two-band visible source from the static base albedo instead of the incidence-angle albedo the boundary condition applied. This reinstates a known energy leak at grazing incidence and announces itself loudly; for A/B comparison only. | [internal] |
| `PTM_LEGACY_ALBEDO_NORMAL_A0` | 0 | Vestigial. With the per-face single-scattering albedo now in use, the Hapke normal-incidence short-circuit returns the same value as the continuous branch, so this changes nothing. Kept as a labeled no-op; it announces itself when set. | [internal] |
| `PTM_LEGACY_FIXED_SOLARCONST` | 0 | `=1` restores the fixed 1 AU solar constant in place of the per-timestep instantaneous value. Byte-identical to the pre-v1.0.3 path. | [internal] |
| `PTM_HMATRIX_VF_REPLACE_GPU` / `PTM_HMATRIX_VF_REPLACE_CPU` | unset (the config decides) | Override which radiosity solver `PTM_HMATRIX_VF_REPLACE` hands the replacement matrix to. **Neither is needed any more:** the route follows `radiation.use_gpu`, and whichever route engaged is named on stdout (`[PTM_HMATRIX_VF_REPLACE] ROUTE = GPU/CPU (…)`). Previously `_GPU` was a second, separate switch that no tracked launcher set, so a config asking for GPU radiosity ran the whole radiosity solve on the CPU. Setting **both** is FATAL (they select opposite routes; the engine refuses to guess). ⚠️ Turning the GPU route ON is a **results change**, not a speed-only one: measured against the same run's CPU route, worst temperature max\|Δ\| = **6.6 mK** on the A3GT lunar polar site A matrix (weak coupling, max ΣF = 0.2) but **0.27 K** on the 509k moon3d flagship (strong coupling), with a 99.6–99.9 % one-sided sign census — truncation and float32-vs-float64, not symmetric round-off. The matrix is always installed as the CPU operator first and only *promoted* in `initGPU()`, so declining the GPU costs nothing and falls back to the production CPU solver; see [radiation-exchange.md §6.5](radiation-exchange.md). | [experimental] |
| `PTM_RADIOSITY_CHUNK_NNZ` | auto (2.0 GB per buffer) | Lower the per-chunk stored-entry cap for the chunked GPU sparse product. The automatic budget leaves margin under the 2 GB single-allocation wall; lower it only to reproduce a chunk split. | [internal] |
| `PTM_RAD_FORCE_SINGLE_FAIL`, `PTM_RAD_FORCE_POOL_FAIL` | unset | Fault injection: force the single-GPU or multi-GPU radiosity initialization to fail so the fallback path can be exercised. Testing only. | [internal] |
| `PTM_RAD_GPU_CHECK_INTERVAL` | unset (auto) | Pin the GPU radiosity Jacobi's convergence-check interval, overriding both `RadiationConfig::convergence_check_interval` and the automatic per-matrix rule (`resolveConvergenceCheckInterval`: check every iteration above ≈7.6e7 stored non-zeros, every 10 below — see [radiation-exchange.md §6.4](radiation-exchange.md)). This changes how many Jacobi iterations run, so it **changes results** unless the solve has already stagnated; the override announces itself on startup. Exists so the auto rule can be falsified at a new scale without a rebuild. | [internal] |
| `PTM_SOLAR_ALBEDO` | unset | Bond albedo for the reflected-solar radiosity source (e.g. `0.12` for the Moon); without it the solar source scatters with the thermal reflectivity 1−ε (~0.05), under-scattering sunlight. Also feeds `PTM_BEAM_SOLAR`. | [standard] (set with TWO_BAND) |
| `PTM_RAD_EMISSIVITY` | unset | Forces a uniform radiosity emissivity (consistency studies). | [internal] |
| `PTM_RAD_ITER_TRACE` | unset | Per-Jacobi-sweep trace of the radiosity solve, one stderr line per sweep: `dB_abs` (max\|B_new−B_old\| in W/m², which is what the **GPU** criterion tests), `B_max`, and their ratio `rel` (which is what the **CPU** criterion tests). The two paths stop on different quantities from the same configured number, so this is the way to see both at once. **Changes GPU behavior while set**: it forces a convergence check every iteration, overriding the interval `resolveConvergenceCheckInterval` resolved for that matrix (otherwise the GPU can only stop at a multiple of it), so each solve stops on the iteration it converges and the counts are not shipped cost. The override announces itself next to the interval line; use `PTM_RAD_COST_REPORT` alone to measure shipped cost. Also emits the `[rad-cost]` summary. | [internal] |
| `PTM_RAD_ITER_TRACE_FULL` | unset | With `PTM_RAD_ITER_TRACE`: also suppress the early exit and run every solve to `max_iterations`, exposing the decay curve past the shipped stop. This is how you measure what the retired sweeps were actually worth. Diagnostic runs only — it changes results. | [internal] |
| `PTM_RAD_GPU_FUSE_BANDS` | 1 (ON) | The GPU band-separated radiosity solve reads each view-factor row ONCE and advances both bands (the CPU solver has done this since Tier 2). `=0` restores the legacy route: two separate solves over the same matrix, roughly twice the matrix traversals for the same answer. Per band the arithmetic is unchanged, so the two routes are **bit-identical** — asserted by `test_two_band_rownorm` section B1f, which compares both bands exactly rather than to a tolerance. Turning it off announces itself. | [kill-switch] |
| `PTM_RAD_GPU_ABS_CRITERION` | unset | `=1` restores the legacy GPU stopping test, max\|dB\| < max(tol, 1e-5) W/m², in place of the CPU's relative max\|dB\|/max\|B\| < tol. Same configured number, two different questions: at B ≈ 900 W/m² the absolute bar is ~900× stricter, so the GPU ran strictly more Jacobi sweeps than the CPU for the same tolerance. Diagnostic/bisect only; it announces itself. | [kill-switch] |
| `PTM_RAD_COST_REPORT` | unset | One `[rad-cost]` line at exit: radiosity solves, total view-factor-matrix traversals, and total in-solver wall time. Does **not** perturb the solve, so this is the honest measure of radiosity cost — whole-run wall clock cannot separate radiosity from illumination and thermal on a shared machine. **Scope:** the counters are charged in `RadiositySolver::solve()`, so a multi-timestep `solveBatch()` that dispatches straight to the multi-GPU pool (never going through `solve()`) contributes nothing to `solves`/`wall`; single-timestep solves, the serial batch fallback, and every CPU path are counted. | [internal] |
| `PTM_BEAM_SOLAR` | 0 | Smith-style beamed reflected-solar emission on wall→floor pairs. | [experimental] |
| `PTM_RAD_COUPLING_RELAX` | 1.0 | Picard under-relaxation ω for the thermal-radiation coupling loop; safety knob for pathological meshes. | [tuning] |
| `PTM_COUPLING_MODE` | unset (config wins; config default `bounded`) | Escape hatch. Overrides `radiation.coupling_mode` = `bounded` \| `fixed` \| `legacy` without editing the config, so a run published before the fix can be reproduced exactly. Validated at config time — an unrecognized value is FATAL before the illumination pass, never silently ignored. See `docs/radiation-exchange.md` § "Chunk-decomposition independence". | [standard] |
| `PTM_EQ_INDIRECT_SOURCE` | unset (config wins; config default `auto`) | Escape hatch. Overrides `thermal.eq_indirect_source` = `auto` \| `converged` \| `off` without editing the config, so a run published before the equilibration-indirect default can be reproduced bit-for-bit. `auto` estimates the per-face mean indirect flux from the coupling loop's iteration-0 sweep and adds it to the equilibration forcing; `converged` reads it after the coupling loop converges instead (removes the iteration-0 low bias, costs one extra coupling loop per pass); `off` is the pre-2026-08 behavior. A non-empty `thermal.eq_indirect_flux_file` overrides all three. Validated at startup — an unrecognized value is FATAL, never silently ignored. See `docs/radiation-exchange.md` § 7.7. | [standard] |
| `PTM_EQ_INDIRECT_PASSES` | unset (config wins; config default `1`) | Escape hatch. Overrides `thermal.eq_indirect_passes` — how many times the internal estimate is refreshed and the equilibration repeated. The default `1` is the 2026-08-11 single-pass behavior, bit-for-bit; a second `auto` pass is that estimator's own knee but buys far less per second than `PTM_EQ_INDIRECT_SOURCE=converged` does. Must be ≥ 1; `0` is rejected rather than treated as an alias for `off`. A non-integer value is FATAL. See `docs/radiation-exchange.md` § 7.7. | [standard] |
| `PTM_EXCHANGE_SELECTION` | unset (config wins; config default `all`) | Escape hatch. Overrides `radiation.exchange_selection` = `all` \| `shadow` without editing the config, so a run published before the fix can be reproduced exactly. `shadow` restores the window-scoped shadow-fraction gate that silently zeroes the indirect flux on faces a chunk happens to find sunlit — the run warns and names the count. Validated at config time — an unrecognized value is FATAL before the illumination pass, never silently ignored. See `docs/radiation-exchange.md` § "The radiation-exchange face set". | [standard] |
| `PTM_RAD_RERUN_MAX_GPUS` | 1 | Cap on GPUs used for the selective-face re-run stage. | [tuning] |
| `PTM_RAD_SKIP_EQ_STEPS` | **1 (ON since 2026-08-15)** | The coupling loop's radiosity sweep starts at the first OUTPUT step instead of at t=0, so the equilibration span is not solved. On a two-phase run the whole-mesh GPU solver records surface temperatures over the output window only, so those steps solve the enclosure against a 0 K surface and nothing physical consumes them — measured **46.5 %** off a full 509k lunar chunk's wall clock (4951.9 s → 2650.5 s at 3000 eq + 20 output steps; **32.9 %** on a shorter 600 + 20 arm, sweep phase 459 s → 12 s). Together with the default-on `PTM_RAD_SELECT_OUTPUT_WINDOW` the skip is byte-identical — verified 22/22 `statistics/*.bin` on the 509k mesh (skip+window vs window alone). **`=0` restores the whole-span sweep.** With an EXPLICIT `=1` the engine refuses to run — never degrades quietly — when `thermal.use_gpu` is false, when the run is single-cadence, when there is a location export, or when a coupling iteration falls back to the CPU re-thermal, because the equilibration span is load-bearing on each of those paths. Left at its default the gate instead **stands down on those paths and prints why**; the mid-loop CPU-re-thermal fallback stays FATAL either way, because by then the span really is unsolved (re-run with `=0`). | [kill-switch] |
| `PTM_RAD_SELECT_OUTPUT_WINDOW` | **1 (ON since 2026-08-15), gated** | Restricts the coupling loop's face-selection max-indirect reduction to the output window `[equilibration_steps, total)`. The selector feeds that maximum to a linearized **output-window** temperature response, but before this fix reduced it over all t, so on a two-phase run it could be set by the equilibration span — where the enclosure is at 0 K and only reflected sunlight is present. Measured on the 509k lunar flagship: the equilibration span held the maximum on **35.5 %** of faces, and froze 395 faces at sweep 5 where the output-window selector freezes 155. ⚠️ **The default only applies where that justification holds — the two-phase GPU re-thermal.** On a single-cadence run or the CPU re-thermal the solver records every step, the equilibration span carries real temperatures and real inter-face IR, and the re-thermal re-integrates it, so clipping the reduction would UNDER-select; there the gate stands down and says so. (Measured on the 12,800-face 80 N crater: **0 of 12,800** faces attain their maximum inside the span on the two-phase GPU path against **2,584 of 12,800** on the CPU coupled path.) An explicit `=1` is still honored on those paths, with a warning. **`=0` restores the pre-2026-08-15 whole-span reduction** everywhere, which on a two-phase GPU run is a results change back to the defect. Costs nothing measurable either way. | [kill-switch] |
| `PTM_RHO_ADVISORY` | 1 (ON) | Before the first coupling sweep, power-iterate the view-factor operator **the solve actually applies** — `diag(1/max(1, Σ_j F_ij))·F`, since the radiosity renormalizes every row of `F·x`, not raw `F` — and report ρ, the implied Picard rate `r = ε·ρ/(1−(1−ε)·ρ)` at the *operative* radiosity emissivity, and an upper bound on the sweeps needed for a 1 K and 0.1 K coupling increment; **WARN** when `radiation.max_coupling_iterations` is below the 1 K count. **Advisory only — nothing in the solve reads it and no number changes.** Coupled runs only (a radiation-off run never reaches it) and skipped when `max_coupling_iterations < 2`. Costs one matvec per iteration, ≤ 60 iterations — measured 3.0 s at 495M nnz on this machine (48-core OpenMP matvec), well under 1 % of one coupling sweep on that mesh. **`=0` skips it** and says so. | [kill-switch] |
| `PTM_HMATRIX_VF_REPLACE` | unset | Replace the stored view-factor matrix with an independently computed reference (a sparse-matrix file whose every entry passed an explicit shadowing test; row sums ≤ 1, reciprocity preserved). The program stops with a clear error rather than run with a mismatched or invalid correction. It routes the matrix to the GPU or CPU radiosity operator according to **`radiation.use_gpu`**; see `PTM_HMATRIX_VF_REPLACE_GPU`/`_CPU` above for the overrides and for what happens when the GPU declines. | [experimental] |
| `PTM_HMATRIX_ROWFIX` | unset | Row-sum correction for over-counted view factors: a symmetric rescaling of the matrix (D·F·D), iterated (Sinkhorn algorithm) until each face's total view factor matches a reference value supplied as a file of n double-precision numbers. If the iteration fails to converge the program stops rather than run with an invalid matrix. | [experimental] |
| `PTM_HMATRIX_ROWFIX_MODE` | `damp` | `damp` = reductions only (stable, validated); `full` = also restores under-coupling but **diverges on this problem class** — the convergence check stops the run. Use `damp`. | [experimental] |
| `PTM_HMATRIX_ROWFIX_RCAP` | 4.0 | FULL-mode cap: restoration ≤ RCAP×stored per row. | [experimental] |
| `PTM_HMATRIX_ROWFIX_DMAX` | 4.0 | Clamp on the congruence diagonal d. | [experimental] |
| `PTM_HMATRIX_ROWFIX_RELAX` | 0.5 (damp) / 0.25 (full) | Sinkhorn under-relaxation. | [experimental] |
| `PTM_HMATRIX_ROWFIX_TOL` | 1e-2 | Convergence gate tolerance; loosening it is explicitly not recommended. | [experimental] |

## H-matrix view-factor build

| Variable | Default | Effect | Status |
|---|---|---|---|
| `PTM_HMATRIX_THREADS` | auto | Thread count for the view-factor (H-matrix) computation. Cap it on large meshes to avoid running out of memory during the build (some blocks fall back to dense storage and are memory-hungry). | [tuning] |
| `PTM_HMATRIX_ACA_CAP` | 12000 | Max faces/side for ACA low-rank blocks (0 = off); larger admissible blocks are zeroed or subdivided. | [tuning] |
| `PTM_HMATRIX_SUBDIVIDE` | config (default on) | 1/0 override of `radiation.hmatrix_subdivide_admissible`: bisect and re-run ACA on oversized admissible blocks instead of zeroing them. Zeroing them removes crater floor-to-wall coupling above roughly 100k faces. | [tuning] |
| `PTM_HMATRIX_HIER_LEVELS` | mesh-size clamp | Override the number of octree subdivision levels (a memory lever: more levels = smaller per-block working memory). | [tuning] |
| `PTM_HMATRIX_SPARSE_CAP` | 5000 (out-of-core) / 300 | Cap on per-block sparse-fallback size. | [tuning] |
| `PTM_HMATRIX_RAY_LIFT` | 0.0 | Lift visibility-ray origins along the face normal by this fraction of the local edge (clears sub-mesh roughness on grazing sightlines). | [experimental] |
| `PTM_HMATRIX_HORIZON_VIS` | 0 | Use heightfield-horizon visibility instead of thin Embree rays in the VF build. | [experimental] |
| `PTM_HMATRIX_SMOOTH_NORMALS` | 0 | Use locally-averaged (macroscopic) normals for the VF cosine terms on fine rough meshes. | [experimental] |
| `PTM_HMATRIX_SMOOTH_RADIUS` | 300 (m) | Averaging radius for SMOOTH_NORMALS. | [experimental] |

## Memory and GPU dispatch

| Variable | Default | Effect | Status |
|---|---|---|---|
| `PTM_RAM_ILLUM_MAX_GB` | 10 | Threshold (GB) above which the illumination time series is streamed from the chunked store rather than held entirely in memory. Raise on high-memory machines, lower on laptops. It selects the FORCING SOURCE only: above the threshold the CPU run still uses the configured integrator (SIMD batch by default, fed one 8-face tile at a time; `--no-simd-batch` still gets the scalar one). It used to also swap the solver to CPU scalar. The thermal field is identical by construction — same solver, composer, seed and boundary conditions — and was measured byte-identical over the whole run directory on the 12,800-face 80N crater and an 81,920-face lunar ico6; the illumination statistics take a separate streamed reduction, which those two comparisons also covered but which is not guaranteed in general. `[thermal-solver] PATH:` in the log and `thermal_solver_core` in `provenance.txt` state which integrator actually ran. | [tuning] |
| `PTM_MAX_TIMESTEPS_PER_DISPATCH` | machine profile, else 300 | Ceiling on how many thermal timesteps one GPU work submission may contain (protection against GPU driver timeouts); normally supplied by `--calibrate-gpu`. Setting it explicitly also disables the inline auto-calibration probe. | [tuning] |
| `PTM_NO_AUTO_CALIBRATE` | unset | Disables the inline GPU auto-calibration probe (`thermal.auto_calibrate_gpu`, default on), restoring the machine-profile or conservative dispatch ceiling. Affects dispatch batching only, never the physics. | [tuning] |
| `PTM_LOG_SUBDISPATCH` | off | Log the timing of each GPU work submission (diagnostics for GPU driver hangs). | [internal] |
| `PTM_GPU_MIN_VRAM_MARGIN_GIB` | config `gpu.min_vram_margin_gib` (2.0) | Floor on the per-device VRAM headroom a resident radiosity CSR replica may leave behind, in GiB. Overrides the config key, announces the override, and makes the floor **enforcing on every platform** (the config default enforces only on macOS+AMD and is advisory elsewhere). A malformed value warns and is ignored. `0` = fit check only, the pre-2026-08-22 behavior. See [gpu-wedge-guardrails.md](gpu-wedge-guardrails.md). | [tuning] |
| `PTM_RAD_RERUN_MAX_GPUS` | configured thermal device count | Lowers the device count the selected-faces radiation-coupling re-thermal may use. `=1` genuinely serializes it on one device. Since 2026-08-22 the default no longer caps: the shared re-thermal pool is hoisted once at the configured width instead of being rebuilt every coupling iteration. A non-positive value warns and is ignored. | [tuning] |
| `PTM_GPU_FORCE_WEDGE_PLATFORM` | unset | Test hook for the platform gate: `=1` treats the host as macOS+AMD (guardrails ENFORCE), `=0` treats it as not wedge-prone (guardrails ADVISORY). Announces itself on stdout. Use it to exercise both branches of the VRAM-margin and placement policies on one machine. | [internal] |
| `PTM_GPU_POWER_PROBE_FILE` | unset | Test hook for the macOS power-settings advisory: read this file instead of running `pmset -g custom`. Announces itself. | [internal] |

## Diagnostics and internal tooling

| Variable | Default | Effect | Status |
|---|---|---|---|
| `PTM_DUMP_EQ_SEED` | unset | Write a per-face CSV of the CPU batch path's equilibration seed (`avg_flux`, `T_surf_est`, `T_deep - T_surf_est`) and of the analytic clear-sky forcing that drove the equilibration (`eq_forcing_mean/max`), plus each face's final surface temperature. Written once at the end of the batch solve; unset = no cost and byte-identical output. | [internal] |
| `PTM_DUMP_VF_ROWSUMS` | unset | Write each face's total view factor ΣF to the given path (binary: a 4-byte count, then n double-precision values). | [internal] |
| `PTM_DUMP_QIND_SERIES` | unset | Write the per-face indirect-flux time series (binary, 'QIDT' marker; single-precision values, all timesteps of face 0, then face 1, ...). | [internal] |
| `PTM_DUMP_REFLIR` | unset | Write each face's peak temperature and the indirect flux at that moment (binary, 'RFIR' marker). | [internal] |
| `PTM_VF_ROBUST`, `PTM_VF_NLIFT`, `PTM_VF_MSAMPLE`, `PTM_VF_IGNORE_NEAR`, `PTM_VF_MC_LIFT` | unset | Settings for the standalone view-factor checking tools (`vf_face_probe`, `vf_hemisphere_mc`), which recompute individual view factors by independent Monte Carlo ray tracing for comparison. | [internal] |
| `PTM_PRIMARY_DISK_ALWAYS_SAMPLE` | 0 | `=1` disables the convex-body short-circuit in the finite primary-disk visibility probe, so every face pays the full sample count. Validation use. | [internal] |
| `PTM_PRIMARY_DISK_STATS` | 0 | `=1` prints a census of which primary-disk branch each face took (below horizon, fully lit, dark, sampled). | [internal] |
| `PTM_ESC_BENCH`, `PTM_ESC_DUMP` | unset | Benchmark timing and matrix dump for the GPU sparse product escalation path. | [internal] |
| `PTM_PP_RING_PLY` | unset | View-factor diagnostic pipeline (`scripts/ptm12_diag/`). | [internal] |
| `PTM_RADIANCE_PRUNE_EPS` | unset (off) | Radiance-weighted view-factor pruning threshold (W/m²) applied at VF **assembly** (`scripts/ptm12_diag/assemble_vf_haworth.py`): keep partner j for face i only if F_ij·R_j ≥ ε. Also exposed as the config key `radiation.radiance_prune_eps` (see [configuration.md](configuration.md)); when both are set the **env var takes precedence** and the runners propagate the config value into it only when it is unset. The sim does **not** prune — it logs the effective ε at radiation init. `0`/unset = off (numerically identical to the last digit). See [radiation-exchange.md §10.9](radiation-exchange.md). | [experimental] |
| `PTM_EMBREE_VERBOSE` | unset | `=1` follows the one-line `[Embree]` ISA banner with Embree's own verbose device report (which SIMD targets were compiled into the library and which this CPU supports). The banner is unconditional and names the traversal-kernel tier; this is the definitive long form behind it, and the place to look when the banner reports `cannot determine (built without ray packets)`. It goes to stdout because Embree writes its block there. Diagnostic only — it creates a throwaway device, casts no ray, and changes no result. See [../INSTALL.md](../INSTALL.md) "Embree ISA (performance)". | [internal] |
| `PTM_A3GT_REPO`, `PTM_A3GT_MESH_DIR`, `PTM_A3GT_TEMPLATE_RAD` | unset | Path plumbing for the multi-site batch orchestration scripts (`scripts/a3gt_*`). | [internal] |

## See also

- [../INSTALL.md](../INSTALL.md) — build + SPICE kernel data
- [configuration.md](configuration.md) — the YAML configuration reference
  (config keys, not env vars)
- [gpu-wedge-guardrails.md](gpu-wedge-guardrails.md) — what the GPU guardrail
  variables above are guarding against, and which platform each is active on
- [USER_GUIDE.md](USER_GUIDE.md) — CLI reference and output formats
- [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) — known caveats
