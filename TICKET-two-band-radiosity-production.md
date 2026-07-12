# TICKET — Band-separated (visible + thermal-IR) radiosity: production completion

**Owner:** Ada (impl) · Fermi (review) · Iris (coord) · Paul (decisions)
**Status:** SCOPED, awaiting Paul's go. Prepare-only until validation passes + explicit approval.
**Provenance:** grows out of the 2026-07-10/11 Ingersoll bowl-crater investigation.
Refs: `docs/two-band-radiosity-derivation.{tex,pdf}` (spec), `SESSION-NOTES-2026-07-11-two-band-radiosity-implement.md`,
`BUGS.md` PTM-07/PTM-08, memory `ingersoll-fsky-accuracy-2026-07-10`.

## Goal
Take the two-band radiosity from "CPU-implemented, gated OFF, confirmed on the uniform-albedo bowl"
to "correct on real bodies, GPU-performant, validated, ready to default-flip" — then re-validate
(bowl + Moon3D + Diviner). Also delivers a new per-run product: **separate absorbed indirect solar
and IR fluxes**.

## Current state (done, prepare-only)
CPU `RadiositySolver::solveTwoBand()` in `radiation_exchange.cpp` (+120/−7), env-gated OFF
(`PTM_RAD_TWO_BAND=1` + `PTM_SOLAR_ALBEDO=A`). Confirmed on the bowl: biases vanish (flat −0.056%
vs Ingersoll Eq 8 across A=0.06–0.30), gray reduction at A=1−ε, energy closes to 1e-14, cost ≈ gray.
Fermi review = **GO-WITH-FIXES**; two correctness gaps were *dormant* in that uniform-albedo test.

## Containment ("not a can of worms") — what does NOT change
View-factor matrix + construction, Embree ray tracing, H-matrix/CSR storage, Crank-Nicolson thermal
solver, illumination/shadowing, config schema (until D1). Gray solver recovered exactly at A=1−ε ⇒
default-OFF path stays byte-identical. The change is confined to: radiosity-module coefficients + a
second solve, the BC's per-face albedo sourcing, an OpenCL kernel mirror, and additive outputs.

---

## Work items

### Phase A — Correctness (must-fix before default-flip)
- **A0 (decisive cheap test, do FIRST).** Two-band on a concave mesh with `Σ_j F_ij > 1` (`vf_row_norm_`
  active) × Europa albedo A≈0.5 × Gauss-Seidel vs Jacobi. Pins PTM-08's real severity and confirms the
  A1 fix restores contraction + energy closure. Not a production run — a small concave sub-mesh.
- **A1 — PTM-08: GS row-norm for the visible band.** `solveGaussSeidel` sums `Σ_j F_ij B_j` raw
  (`radiation_exchange.cpp:762-767`) with no `vf_row_norm_`, unlike Jacobi (`:691-693`). Harmless at
  IR reflectance 0.05, but the visible band (A up to 0.5) diverges when Σ>2. Fix: apply `vf_row_norm_`
  to the GS sum (or route Σ>1 meshes to Jacobi). Restores the derivation's ρ(AF)<1. Also fixes the
  pre-existing gray/IR GS gap.
- **A2 — per-face, incidence-consistent albedo.** `solar_reflectivity_` is a uniform env scalar; the BC
  absorbs with per-face angular/Hapke albedo (`ptm_thermal_sim.cpp:3821-3828`). Energy closes only if
  `A_rad = A_BC` per face per timestep. Source `solar_reflectivity_` from the same albedo the BC uses,
  and make it time-updatable (currently assigned once in `initCommon`) when incidence-albedo is active.

### Phase B — Performance / GPU (must-fix before production-completion)
- **B1 — OpenCL/GPU two-band kernel mirror.** Second `solar_source` (A·Q_solar), second Jacobi loop
  reusing `spmv_kernel` with reflectance A, extra `indirect_flux` reduction with absorptivity (1−A),
  summed into the returned flux; re-enable the multi-GPU pool for two-band (`initGPU:173` currently
  forces CPU). CPU H-matrix Jacobi already works and is PTM-08-protected — only the OpenCL kernel is
  missing. **Biggest single piece; touches the AMD path.**
- **B2 — cost/cache.** The visible cache misses every call in the production nested loop
  (`for iter { for t }`) and pays an O(n) compare+copy. Either remove it or key it correctly; the real
  cost defense is 1–2 SpMV visible convergence (Moon) → low-single-digit %. Instrument hit/miss +
  `picard_s` in a real coupling run to confirm.
- **B3 — thread-safety guard** on the new mutable cache members (assert single-threaded per instance,
  or make the cache call-local) before any concurrency is reintroduced.

### Phase C — Products
- **C1 — per-band output plumbing.** Wire `lastIndirectVisible()`/`lastIndirectIR()` through the manager
  + statistics output so absorbed indirect solar and IR are separate per-face products each run.

### Phase D — Config / default
- **D1 — promote from env to config.** e.g. `radiation.band_separated: true`, albedo from the existing
  albedo config; gray still recoverable.
- **D2 — default-flip.** Paul's call, gated on Phase E. Separate reviewed commit; default stays OFF until then.

### Phase E — Validation battery (Paul's step 3)
- **E1 — bowl Eq 8** re-confirm on final code (regression fixture).
- **E2 — Moon3D do-no-harm:** default-OFF byte-identical to baseline after `ptm_thermal_sim` rebuild
  (current binary links pre-change `radiation.a`). Then default-ON physics run, characterize the change.
- **E3 — Diviner cross-validation (the science payoff):** does correct solar scattering reduce the
  documented PSR 10–20 K cold bias? Compare gray vs two-band PSR temps vs Diviner. *This is the test
  that justifies the effort.* Region = manuscript **Fig. 8 = Haworth SE rim** (confirmed against
  `Moon3D_Paper.pdf` lines 436-446; internal site key **`sunlit`**, clat/clon −88.181°/6.041°).
  RESOLVED baseline (2026-07-11 scout):
  - config `examples/divval_sunlit_fix.yaml` (ring 25→200 km, eq=25 seasonal — the `RUN_OVERRIDE`
    target matching the seasonal SE-rim figure); mesh `data/dem/validation/meshes/sunlit_radcore_r25.ply`
    (99,511 f) + `sunlit_ring_25_200.ply` (954,404 f) ≈ **1.05 M faces**.
  - Diviner data AUTHORITATIVE: `data/diviner_seasonal/polar_south_80_{summer,winter}_{max,avg}_polar.npy`
    (Williams 2019).
  - ⚠️ **VF does NOT exist** (`sunlit_radcore_r25.vf` absent) → must REBUILD for a ~1.05 M-face core+ring
    lunar mesh — the documented GPU-wedge-trigger profile. Config sets `use_hmatrix: true`, so CONFIRM the
    rebuild produces a COMPRESSED H-matrix VF (compact, like the 7 MB A3GT baseline), NOT a multi-GB dense
    one; build it DETACHED + monitored. Do not reuse the evicted 5.9–6 GB deep-Haworth VFs (wrong mesh).
  - ⚠️ Baseline run output `output/divval_sunlit_fix/` and `sunlit_qind.bin` are GONE (evicted) → E3 = two
    fresh full-seasonal runs (gray baseline + two-band).
  - ⚠️ **Albedo gate:** YAML has `incidence_angle_albedo: true`, but that flipped meaning on 2026-07-02
    (`e9c6946`) — it was a dead no-op (→ Hayne 2017 angular) when the figure was made, now LIVE (→ Hapke).
    Set **`incidence_angle_albedo: false`** to reproduce the original figure's albedo treatment.
  - OPEN confirmations for Paul before E3 runs: config `divval_sunlit_fix` (ring 25→200, eq=25) vs the
    alt `divval_sunlit_cr` (ring 25→40, eq=2); and mesh core r25 vs r32.
- **E4 — CPU/GPU + N=1 vs N=4 GPU equivalence** for two-band (after B1).
- **E5 — full `ptm-validate` suite.**

### Phase F — Housekeeping
- **F1** amend `BUGS.md` (PTM-08 visible-band elevation; PTM-07 gates). **F2** fresh branch
  `feat/two-band-radiosity`, explicit stage list (never `git add -A`). **F3** session notes + memory.

---

## Recommended sequence & the key decision
The science payoff (E3) does not require the GPU kernel. **De-risk by proving the science before the
big GPU investment:**

1. **A0 → A1 → A2** (correctness; small–medium, contained).
2. **E3 on CPU** at a validation-scale mesh — does two-band reduce the PSR cold bias vs Diviner?
   → If **yes**, the effort is justified → proceed to **B1 (GPU)** + full E + D.
   → If **no / marginal**, we've spent little, learned the PSR bias is elsewhere, and can leave two-band
     as a correctness/energy-conservation fix (still worth it for Europa) without the GPU lift.
3. **B1 (GPU) → C1 → D1 → E1/E2/E4/E5 → D2 (flip)** only after E3 says go.

**Decision for Paul:** (i) approve this A→E3-gate→B sequence, or do the full build straight through?
(ii) GPU kernel B1 now, or defer until E3 confirms the science? (iii) validation-mesh choice for E3
(which lunar PSR region + Diviner baseline — resolve from the Project-map registry, confirm the path).

## Fermi re-review verdict (2026-07-11) — GO-WITH-FIXES for Phase E
Commit `f5c455e` reviewed. Core PROVEN sound: A1 GS row-norm fix correct (bit-exact where Σ≤1),
`solveTwoBand` matches Eqs 4-6 exactly, GPU gate sound (Σ>1 or H-matrix → protected CPU), cache
removal strictly-more-correct, test fixture is a sound falsifier. The "fixes" are **run-conditions, not
code blockers**:
- **E3 FRAMING CORRECTION (important):** do NOT frame E3 as "does two-band close the ~20 K PSR cold
  bias." Per prior crew findings ([[psr_cold_bias_equilibration]], [[vf_psr_underradiation_todo]]) that
  bias is **IR/equilibration-duty-cycle-dominated**, which two-band does NOT touch. Two-band corrects
  the *reflected-visible* (secondary) channel → expect a **few-K PSR warming, not 20 K**. Frame E3 as
  do-no-harm + quantify the modest two-band warming. [Fermi PREDICTED, E3 measures it.]
- **Deferred per-timestep incidence-albedo update is NOT an E3 blocker** — static-base-albedo mis-states
  the reflected-solar source by (A_Hayne(θ)−A_static)·Q; sign is monotone-cold, magnitude ~0.3–1 K
  (annual flux-weighting), and it warns loudly. Keep `incidence_angle_albedo:false` (Hayne BC); do NOT
  flatten the BC albedo. Optional cheap bracket: run E3 two-band at base vs flux-weighted-grazing flat
  albedo (~0.20); floor-T delta bounds the approximation (<1 K predicted).
- **Byte-identity is CONDITIONAL:** A1 (`radiation_exchange.cpp:890`) is ungated on `two_band_`, so gray
  sparse-GS on Σ>1 meshes now legitimately differs (that IS the PTM-08 fix). Our targets are PREDICTED
  safe (bowl 0 Σ>1; Moon3D do-no-harm radiation-OFF; Haworth H-matrix→Jacobi already had row-norm).
  CONFIRM by grepping each Phase-E run log for the VF-path banner + "rows>1 renormalized" count.
- **GPU two-band won't engage in production** (setSolarAlbedo/enableTwoBand called after initGPU,
  `ptm_thermal_sim.cpp:10620`) → CPU-safe but Haworth (H-matrix) radiosity is CPU regardless; budget
  wall-clock accordingly. Fixable ordering issue (matters only for sparse-Σ≤1 global runs).
- MINOR: 2.5× radiosity (visible solve not hoisted in the for-iter{for-t} nest); per-band stats-file
  output (C1) computed but not persisted yet. Neither gates E3.

## Effort / risk
- A1 small · A2 medium · B1 large (OpenCL, AMD path) · B2/B3/C1/D1 small–medium · E mostly detached compute.
- Highest risk: B1 (GPU kernel on the wedge-prone AMD path) and A2 (per-face albedo threading + time-update).
- Lowest-hanging, high-value: A1 (also fixes a latent pre-existing bug) and the E3 science test.
