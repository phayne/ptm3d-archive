# PTM3D environment variables

Every `PTM_*` environment variable read by the C++ binaries and the shipped
scripts. All are **unset by default**, and unset always reproduces the
established baseline behavior (numerically identical to the last digit wherever a
reference baseline exists). To re-audit this table against the tree, grep `src/`
and `scripts/` for `getenv("PTM_` and compare the result against the names below.

Status tags:

- **[recommended-ON]** — validated fixes that default OFF only so historical
  baseline runs can still be reproduced digit-for-digit. **The validated set is:**
  `PTM_ILLUM_TRUE_NORMALS=1 PTM_ILLUM_RISE_EPS=1 PTM_ILLUM_RISE_KEPS=0.2
  PTM_ILLUM_RISE_HABS=0 PTM_ILLUM_SEAL_HOLES=1 PTM_EQ_PERFACE_FLUX=1`.
  New science runs should set these.
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
| `PTM_SPICE_KERNELS` | unset | Overrides the SPICE kernel directory when none was set by config/CLI (resolution: explicit setting → this env → compiled-in `SPICE_KERNELS_DIR`, default `<repo>/data/kernels`). If no directory resolves, the program stops with a clear error rather than continuing silently. | [standard] |
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
| `PTM_HMATRIX_VF_REPLACE_GPU` | unset | Used with `PTM_HMATRIX_VF_REPLACE`: routes the replacement matrix through the GPU sparse path instead of the CPU operator. | [experimental] |
| `PTM_RADIOSITY_CHUNK_NNZ` | auto (2.0 GB per buffer) | Lower the per-chunk stored-entry cap for the chunked GPU sparse product. The automatic budget leaves margin under the 2 GB single-allocation wall; lower it only to reproduce a chunk split. | [internal] |
| `PTM_RAD_FORCE_SINGLE_FAIL`, `PTM_RAD_FORCE_POOL_FAIL` | unset | Fault injection: force the single-GPU or multi-GPU radiosity initialization to fail so the fallback path can be exercised. Testing only. | [internal] |
| `PTM_SOLAR_ALBEDO` | unset | Bond albedo for the reflected-solar radiosity source (e.g. `0.12` for the Moon); without it the solar source scatters with the thermal reflectivity 1−ε (~0.05), under-scattering sunlight. Also feeds `PTM_BEAM_SOLAR`. | [standard] (set with TWO_BAND) |
| `PTM_RAD_EMISSIVITY` | unset | Forces a uniform radiosity emissivity (consistency studies). | [internal] |
| `PTM_BEAM_SOLAR` | 0 | Smith-style beamed reflected-solar emission on wall→floor pairs. | [experimental] |
| `PTM_RAD_COUPLING_RELAX` | 1.0 | Picard under-relaxation ω for the thermal-radiation coupling loop; safety knob for pathological meshes. | [tuning] |
| `PTM_COUPLING_MODE` | unset (config wins; config default `bounded`) | Escape hatch. Overrides `radiation.coupling_mode` = `bounded` \| `fixed` \| `legacy` without editing the config, so a run published before the fix can be reproduced exactly. Validated at config time — an unrecognized value is FATAL before the illumination pass, never silently ignored. See `docs/radiation-exchange.md` § "Chunk-decomposition independence". | [standard] |
| `PTM_EQ_INDIRECT_SOURCE` | unset (config wins; config default `auto`) | Escape hatch. Overrides `thermal.eq_indirect_source` = `auto` \| `converged` \| `off` without editing the config, so a run published before the equilibration-indirect default can be reproduced bit-for-bit. `auto` estimates the per-face mean indirect flux from the coupling loop's iteration-0 sweep and adds it to the equilibration forcing; `converged` reads it after the coupling loop converges instead (removes the iteration-0 low bias, costs one extra coupling loop per pass); `off` is the pre-2026-08 behavior. A non-empty `thermal.eq_indirect_flux_file` overrides all three. Validated at startup — an unrecognized value is FATAL, never silently ignored. See `docs/radiation-exchange.md` § 7.7. | [standard] |
| `PTM_EQ_INDIRECT_PASSES` | unset (config wins; config default `1`) | Escape hatch. Overrides `thermal.eq_indirect_passes` — how many times the internal estimate is refreshed and the equilibration repeated. The default `1` is the 2026-08-11 single-pass behavior, bit-for-bit; a second `auto` pass is that estimator's own knee but buys far less per second than `PTM_EQ_INDIRECT_SOURCE=converged` does. Must be ≥ 1; `0` is rejected rather than treated as an alias for `off`. A non-integer value is FATAL. See `docs/radiation-exchange.md` § 7.7. | [standard] |
| `PTM_EXCHANGE_SELECTION` | unset (config wins; config default `all`) | Escape hatch. Overrides `radiation.exchange_selection` = `all` \| `shadow` without editing the config, so a run published before the fix can be reproduced exactly. `shadow` restores the window-scoped shadow-fraction gate that silently zeroes the indirect flux on faces a chunk happens to find sunlit — the run warns and names the count. Validated at config time — an unrecognized value is FATAL before the illumination pass, never silently ignored. See `docs/radiation-exchange.md` § "The radiation-exchange face set". | [standard] |
| `PTM_RAD_RERUN_MAX_GPUS` | 1 | Cap on GPUs used for the selective-face re-run stage. | [tuning] |
| `PTM_RAD_SKIP_EQ_STEPS` | 0 | `=1` starts the coupling loop's radiosity sweep at the first OUTPUT step instead of at t=0, so the equilibration span is not solved. On a two-phase run the whole-mesh GPU solver records surface temperatures over the output window only, so those steps solve the enclosure against a 0 K surface and nothing physical consumes them — measured **32.9 %** off a 509k lunar chunk's wall clock (sweep phase 459 s → 12 s). It is **not** byte-identical to the shipped engine on its own: the face selector's max-indirect reduction runs over all t, and the difference is 6 of 22 `statistics/*.bin`, 33 of 509,337 faces, ≤ 2.5e-5 K. Set together with `PTM_RAD_SELECT_OUTPUT_WINDOW=1` and it is byte-identical (verified 22/22 on the 509k mesh). Refuses to run — never degrades quietly — when `thermal.use_gpu` is false, when the run is single-cadence, when there is a location export, or when a coupling iteration falls back to the CPU re-thermal, because the equilibration span is load-bearing on each of those paths. | [experimental] |
| `PTM_RAD_SELECT_OUTPUT_WINDOW` | 0 | `=1` restricts the coupling loop's face-selection max-indirect reduction to the output window `[equilibration_steps, total)`. The selector feeds that maximum to a linearized **output-window** temperature response, but reduces it over all t, so on a two-phase run it can be set by the equilibration span — where the enclosure is at 0 K and only reflected sunlight is present. Measured on the 509k lunar flagship: the equilibration span holds the maximum on **35.5 %** of faces, and freezes 395 faces at sweep 5 where the output-window selector freezes 155. A **results change** (it re-thermals faces the shipped selector freezes), hence default off pending a ruling; it costs nothing measurable. | [experimental] |
| `PTM_HMATRIX_VF_REPLACE` | unset | Replace the stored view-factor matrix with an independently computed reference (a sparse-matrix file whose every entry passed an explicit shadowing test; row sums ≤ 1, reciprocity preserved). The program stops with a clear error rather than run with a mismatched or invalid correction. | [experimental] |
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
| `PTM_A3GT_REPO`, `PTM_A3GT_MESH_DIR`, `PTM_A3GT_TEMPLATE_RAD` | unset | Path plumbing for the multi-site batch orchestration scripts (`scripts/a3gt_*`). | [internal] |

## See also

- [../INSTALL.md](../INSTALL.md) — build + SPICE kernel data
- [configuration.md](configuration.md) — the YAML configuration reference
  (config keys, not env vars)
- [USER_GUIDE.md](USER_GUIDE.md) — CLI reference and output formats
- [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) — known caveats
