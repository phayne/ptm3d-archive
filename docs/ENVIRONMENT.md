# PTM3D environment variables

Every `PTM_*` environment variable read by the C++ binaries and the shipped
scripts (exhaustive search of `src/` + `scripts/`, 2026-07-29). All are **unset
by default**, and unset always reproduces the established baseline behavior
(numerically identical to the last digit wherever a reference baseline exists).

Status tags:

- **[recommended-ON]** — validated fixes that default OFF only so historical
  baseline runs can still be reproduced digit-for-digit. **Production-validated
  runs (the Moon3D campaigns) used:**
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
only, and a regression pair could silently run under two different environments
— internal bug PTM-42):

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
| `PTM_ILLUM_TRUE_NORMALS` | off | Use the mesh's true triangle-winding normals for illumination instead of the radial-up flip. The flip inverted ~97% of steep-wall faces on polar terrain, fabricating sunlit overhangs. Validated: removes the false >380 K overhang tail without darkening any genuinely sunlit face. | **[recommended-ON]** `=1` |
| `PTM_ILLUM_RISE_EPS` | off | Enables the out-of-plane-rise self-relief discriminator for shadow rays (PTM-13): separates a face's own mm-scale relief from a real occluding rim by rise height rather than the legacy 1 cm ray offset (which hopped real rims and leaked direct sun into PSRs). | **[recommended-ON]** `=1` |
| `PTM_ILLUM_RISE_KEPS` | 0.2 | Facet-edge fraction defining the edge-proportional self-relief floor for RISE_EPS. | **[recommended]** leave at `0.2` |
| `PTM_ILLUM_RISE_HABS` | unset (edge-proportional) | Absolute self-relief floor override in meters; `0` = exact source-triangle-only exclusion. Validated: `=0` eliminates 100% of the grazing-incidence light leaks without introducing spurious self-shadowing speckle. | **[recommended-ON]** `=0` |
| `PTM_ILLUM_SEAL_HOLES` | off | Fills interior mesh holes with shadow-only faces (never solved/output) so finite-disk penumbra rays cannot thread through mesh gaps (PTM-18 "crater-center flash"). Validated on the Moon3D mesh: 49 shadow-only fills, flash → 0, collateral 1/778. | **[recommended-ON]** `=1` |
| `PTM_ILLUM_RISE_EPS2` | off | PTM-13 "option 3" multi-hit refinement sub-mode (implies RISE_EPS). Superseded by the RISE_EPS + RISE_HABS=0 recipe; did not fix the spurious-illumination classes on its own. | [experimental] |
| `PTM_ILLUM_RISE_BARY_EPS` | internal default | Barycentric tolerance for the RISE_EPS source-triangle exclusion. | [tuning] |
| `PTM_ILLUM_TOPO_SELFSKIP` | off | Alternative PTM-13 mitigation: topological self-skip of the source face's neighborhood (CPU illumination path only; does not engage on the GPU tracer). | [experimental] |
| `PTM_ILLUM_SELFSKIP_RINGS` | 1 | Neighborhood ring count for TOPO_SELFSKIP. | [experimental] |
| `PTM_ILLUM_HORIZON_CULL` | off | Heightfield-horizon pre-cull of shadow rays. | [experimental] |
| `PTM_ILLUM_HORIZON_SLACK_DEG` | internal default | Slack angle for HORIZON_CULL. | [experimental] |
| `PTM_ILLUM_HORIZON_CELL_SCALE` | internal default | Heightfield cell scale for HORIZON_CULL. | [experimental] |
| `PTM_DISABLE_STRICT_HORIZON` | 0 | `=1` disables strict outward-normal enforcement when computing face geometry. Debug escape hatch. | [internal] |

## Thermal equilibration

| Variable | Default | Effect | Status |
|---|---|---|---|
| `PTM_EQ_PERFACE_FLUX` | off | Seeds each column's equilibration from **its own** time-mean absorbed flux + geothermal Qb instead of the legacy first-10-step instantaneous flux with a 100 K nightside floor. PSR floors cold-start at the geothermal steady state (~24.7 K on the Moon) instead of being stamped warm they cannot cool down from within a run. Unset, results are numerically identical to the last digit of the historical baseline; the bowl-crater validation test passes either way. | **[recommended-ON]** `=1` |
| `PTM_EQ_WITH_INDIRECT` | off | Folds the (phase-mapped) indirect radiation flux into re-equilibration for coupled seasonal runs. | [experimental] |
| `PTM_EQ_INDIRECT_CYCLES` | 2 | Number of prior seasonal cycles assumed stationary for the EQ_WITH_INDIRECT phase mapping. | [experimental] |
| `PTM_EQ_RETHERMAL_BATCH` | 4000 | GPU batch cap during EQ_WITH_INDIRECT re-equilibration (the 50k default overruns the macOS/AMD GPU command-buffer watchdog). | [tuning] |

## Radiation exchange / radiosity

| Variable | Default | Effect | Status |
|---|---|---|---|
| `PTM_RAD_TWO_BAND` | off (gray) | Band-separated radiosity: the visible channel scatters with the Bond albedo, the IR channel with 1−ε; absorbed indirect flux = visible + IR. **Requires `PTM_SOLAR_ALBEDO`** (warns loudly and falls back to gray if unset). Used by the two-band production campaigns. | [standard] |
| `PTM_SOLAR_ALBEDO` | unset | Bond albedo for the reflected-solar radiosity source (e.g. `0.12` for the Moon); without it the solar source scatters with the thermal reflectivity 1−ε (~0.05), under-scattering sunlight. Also feeds `PTM_BEAM_SOLAR`. | [standard] (set with TWO_BAND) |
| `PTM_RAD_EMISSIVITY` | unset | Forces a uniform radiosity emissivity (consistency studies). | [internal] |
| `PTM_BEAM_SOLAR` | 0 | Smith-style beamed reflected-solar emission on wall→floor pairs. | [experimental] |
| `PTM_RAD_COUPLING_RELAX` | 1.0 | Picard under-relaxation ω for the thermal-radiation coupling loop; safety knob for pathological meshes. | [tuning] |
| `PTM_RAD_RERUN_MAX_GPUS` | 1 | Cap on GPUs used for the selective-face re-run stage. | [tuning] |
| `PTM_HMATRIX_VF_REPLACE` | unset | Replace the stored view-factor matrix with an independently computed reference (a sparse-matrix file whose every entry passed an explicit shadowing test; row sums ≤ 1, reciprocity preserved). The program stops with a clear error rather than run with a mismatched or invalid correction. Campaign-specific corrected view factors were used in Moon3D/Haworth production. | [experimental] |
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
| `PTM_HMATRIX_SUBDIVIDE` | config | 1/0 override of `subdivide_large_admissible`: bisect + re-ACA oversized admissible blocks instead of zeroing them. | [tuning] |
| `PTM_HMATRIX_HIER_LEVELS` | mesh-size clamp | Override the number of octree subdivision levels (a memory lever: more levels = smaller per-block working memory). | [tuning] |
| `PTM_HMATRIX_SPARSE_CAP` | 5000 (out-of-core) / 300 | Cap on per-block sparse-fallback size. | [tuning] |
| `PTM_HMATRIX_RAY_LIFT` | 0.0 | Lift visibility-ray origins along the face normal by this fraction of the local edge (clears sub-mesh roughness on grazing sightlines). | [experimental] |
| `PTM_HMATRIX_HORIZON_VIS` | 0 | Use heightfield-horizon visibility instead of thin Embree rays in the VF build. | [experimental] |
| `PTM_HMATRIX_SMOOTH_NORMALS` | 0 | Use locally-averaged (macroscopic) normals for the VF cosine terms on fine rough meshes. | [experimental] |
| `PTM_HMATRIX_SMOOTH_RADIUS` | 300 (m) | Averaging radius for SMOOTH_NORMALS. | [experimental] |

## Memory and GPU dispatch

| Variable | Default | Effect | Status |
|---|---|---|---|
| `PTM_RAM_ILLUM_MAX_GB` | 10 | Threshold (GB) above which the illumination time series is read from disk in pieces rather than held entirely in memory. Raise on high-memory machines to keep everything in RAM (faster); lower on laptops. | [tuning] |
| `PTM_MAX_TIMESTEPS_PER_DISPATCH` | machine profile, else 300 | Ceiling on how many thermal timesteps one GPU work submission may contain (protection against GPU driver timeouts); normally supplied by `--calibrate-gpu`. | [tuning] |
| `PTM_LOG_SUBDISPATCH` | off | Log the timing of each GPU work submission (diagnostics for GPU driver hangs). | [internal] |

## Diagnostics and internal tooling

| Variable | Default | Effect | Status |
|---|---|---|---|
| `PTM_DUMP_EQ_SEED` | unset | Write a per-face CSV of the CPU batch path's equilibration seed (`avg_flux`, `T_surf_est`, `T_deep - T_surf_est`) and of the analytic clear-sky forcing that drove the equilibration (`eq_forcing_mean/max`), plus each face's final surface temperature. Written once at the end of the batch solve; unset = no cost and byte-identical output. Added to settle PTM-51's disputed mechanism by measurement. | [internal] |
| `PTM_DUMP_VF_ROWSUMS` | unset | Write each face's total view factor ΣF to the given path (binary: a 4-byte count, then n double-precision values). | [internal] |
| `PTM_DUMP_QIND_SERIES` | unset | Write the per-face indirect-flux time series (binary, 'QIDT' marker; single-precision values, all timesteps of face 0, then face 1, ...). | [internal] |
| `PTM_DUMP_REFLIR` | unset | Write each face's peak temperature and the indirect flux at that moment (binary, 'RFIR' marker). | [internal] |
| `PTM_VF_ROBUST`, `PTM_VF_NLIFT`, `PTM_VF_MSAMPLE`, `PTM_VF_IGNORE_NEAR`, `PTM_VF_MC_LIFT` | unset | Settings for the standalone view-factor checking tools (`vf_face_probe`, `vf_hemisphere_mc`), which recompute individual view factors by independent Monte Carlo ray tracing for comparison. | [internal] |
| `PTM_PP_RING_PLY` | unset | PTM-12 diagnostic pipeline (`scripts/ptm12_diag/`). | [internal] |
| `PTM_RADIANCE_PRUNE_EPS` | unset (off) | Radiance-weighted view-factor pruning threshold (W/m²) applied at VF **assembly** (`scripts/ptm12_diag/assemble_vf_haworth.py`): keep partner j for face i only if F_ij·R_j ≥ ε. Also exposed as the config key `radiation.radiance_prune_eps` (see [configuration.md](configuration.md)); when both are set the **env var takes precedence** and the runners propagate the config value into it only when it is unset. The sim does **not** prune — it logs the effective ε at radiation init. `0`/unset = off (numerically identical to the last digit). See [radiation-exchange.md §10.9](radiation-exchange.md). | [experimental] |
| `PTM_A3GT_REPO`, `PTM_A3GT_MESH_DIR`, `PTM_A3GT_TEMPLATE_RAD` | unset | A3GT campaign-runner plumbing (`scripts/a3gt_*`). | [internal] |

## See also

- [../INSTALL.md](../INSTALL.md) — build + SPICE kernel data
- [configuration.md](configuration.md) — the YAML configuration reference
  (config keys, not env vars)
- [USER_GUIDE.md](USER_GUIDE.md) — CLI reference and output formats
- [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) — known caveats (the PTM-nn bug
  ids above refer to the project's internal bug ledger)
