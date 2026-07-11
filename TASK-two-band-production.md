# TASK — Two-band radiosity: production completion (Ada, prepare-only)

Checkpoint for the TICKET-two-band-radiosity-production.md work. Prepare-only:
NO commit/push/tag, NO default-flip, gray always recoverable + default OFF.
Artifacts in `output/two_band_prod/` (durable, gitignored — never /tmp).

Provenance: HEAD e50770b + uncommitted (branch fix/ema-camera-noise-temp-k, tree dirty).

## Phase ledger (result artifact FIRST, then narrative)

### A0 — decisive severity test (PTM-08 in the regime it bites)  [status: DONE]
`src/cpp/radiation/test_two_band_rownorm.cpp` (tracked). PRE-fix: GS over-couples
+2.2% (gray A=0.05) → +91% (Europa A=0.5, s=1.5) → inf (s=2.5), Jacobi finite.
Artifacts: `output/two_band_prod/a0_gs_vs_jacobi.csv`, `a0_run_pre.log`.

### A1 — PTM-08 fix: vf_row_norm_ on the Gauss-Seidel sum  [status: DONE]
`radiation_exchange.cpp:762-767` GS sum now `*= vf_row_norm_[i]` (mirrors Jacobi).
POST-fix: GS==Jacobi to 1e-12..5e-10 across all 4 A0 configs (`a0_run_post.log`).
Bowl do-no-harm PROVABLE: 0/14720 rows > 1 => vf_row_norm_==1 => A1 is identity
on the bowl (gray GS still −6.26%, two-band GS at +0.01% faceting floor).

### A2 — per-face incidence-consistent albedo  [status: DONE]
Solver `setSolarAlbedo(per-face)` (time-updatable) + manager passthrough
(`radiation_exchange.{cpp,h}`). VERIFIED (`a2_energy_closure.csv`): per-face
A_solver==A_BC closes to 0.00e+00%; uniform env scalar leaks −3.75% on a
non-uniform-albedo case (Fermi's predicted leak).

### B2 — cost/cache  [status: DONE — visible cache REMOVED]
Removed the single-slot visible cache (missed every call in production for iter{for t},
paid O(n) compare+copy, was mutable cross-call state). Visible solve re-run each
call = 1-2 SpMV. Cost check: see `bowl_cost_picard.csv`.

### B3 — cache thread-safety guard  [status: DONE via B2]
Cache removed => no mutable cross-call state. last_indirect_vis_/ir_ documented
non-re-entrant per instance (pool uses one solver/device).

### B1 — GPU/OpenCL two-band kernel mirror + row-norm mirror  [status: DONE (single-GPU); pool: DEFERRED]
`solveGPUTwoBand` reuses the verified gray kernels twice (IR solar=0 + visible
ε←1-A/ρ←A). GATED to Σ≤1 meshes (GPU Jacobi omits row-norm); Σ>1 or pool-requested
=> warn + protected CPU two-band. VERIFIED GPU-vs-CPU on N=64: max rel diff 4.5e-07
(`b1_gpu_vs_cpu.csv`). Multi-GPU pool two-band NOT implemented (falls back to
single-GPU, noticed). No heavy GPU runs (wedge risk) — 64-face correctness only.

### C1 — per-band outputs through manager  [status: DONE (manager getters)]
`RadiationExchangeManager::lastIndirectVisible()/IR()` map per-band absorbed flux to
all faces in `computeIndirectFlux`. Statistics-file plumbing (writing them as
separate per-face products) = remaining (needs the stats-output path; report).

### D1 — radiation.band_separated config path (default OFF)  [status: DONE]
`RadiationConfig.band_separated` + config_parser + solver/manager `enableTwoBand()`.
ptm_thermal_sim sources `face_albedos_rad` + activates CPU two-band when set; WARNS
if incidence-albedo is on (per-timestep update not wired). Default OFF => byte-identical.

### F1 — BUGS.md amend (PTM-08 / PTM-07)  [status: DONE]

### RESUMABLE (clearly scoped, NOT done):
- Multi-GPU pool two-band (single-GPU works; pool falls back w/ notice).
- Per-timestep incidence-albedo update in the ptm_thermal_sim coupling loop (A2
  time-update for the incidence case; static base albedo used now, WARNED).
- C1 statistics-file per-band output (manager getters exist).
- Phase E validation (Fermi/later): bowl+Moon3D do-no-harm byte-identity after
  ptm_thermal_sim rebuild, Diviner, CPU/GPU + N=1 vs N=4, full ptm-validate.

## Gates NOT run by Ada (Fermi + Phase E after review)
E1 bowl Eq8 · E2 Moon3D do-no-harm · E3 Diviner · E4 CPU/GPU + N=1 vs N=4 · E5 full ptm-validate.
