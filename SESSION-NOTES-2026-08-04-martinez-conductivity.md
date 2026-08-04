# SESSION NOTES — Martinez & Siegler (2021) low-temperature conductivity (Ada, 2026-08-04)

Mandate (Sagan, on Paul's concern that PTM3D's conductivity may be inaccurate at the
low temperatures where PSR floors live): read the paper in detail, implement it as a
**selectable, opt-in** conductivity model with the default UNCHANGED, test it, and
re-run the Haworth PSR summer window with it. **PREPARE-ONLY.**

Outcome: steps 1-3 complete and passing. **Step 4 (the Haworth `R_mart` production
run) is deliberately NOT run** — see "Flags for Fermi/Sagan" at the end. The
configuration and driver for it are in place and one command away.

---

## 1. The model, as extracted from the paper

Martinez, A., & Siegler, M. A. (2021), "A Global Thermal Conductivity Model for Lunar
Regolith at Low Temperatures", *JGR Planets* **126**, e2021JE006829,
doi:10.1029/2021JE006829. PDF: `docs/Martinez_lunar-low-temp-conductivity_JGR-Planets_2021.pdf`.

```
Eq. 10:  k(T, rho) = (A1*rho - A2) * k_am(T) + (B1*rho - B2) * T^3
Eq.  9:  k_am(T)   = Ac + Bc*T^-4 + Cc*T^-3 + Dc*T^-2 + Ec*T^-1
                        + Fc*T + Gc*T^2 + Hc*T^3 + Ic*T^4
```

| Ac | Bc | Cc | Dc | Ec | Fc | Gc | Hc | Ic |
|---|---|---|---|---|---|---|---|---|
| -0.203297 | -11.472 | 22.5793 | -14.3084 | 3.41742 | 0.01101 | -2.8049e-5 | 3.35837e-8 | -1.40021e-11 |

`A1 = 5.0821e-6`, `A2 = 5.1e-3`, `B1 = 2.0022e-13`, `B2 = 1.953e-10`. SI units.

`k_am(T)` is the amorphous-endmember solid conductivity of Woods-Robinson et al.
(2019) Eq. 30, a fit to amorphous SiO2 (Touloukian & Ho 1970). The physics of the
change: the **solid** conduction term becomes temperature-dependent, where Watson's
equation (and therefore Hayne et al. 2017, which PTM3D uses) holds it constant.

Full write-up including validity limits, guards and the port's exclusions:
**`docs/martinez-conductivity-model.md`**.

### The finding that matters most for interpreting any run

**This is not a low-temperature-only perturbation on PTM3D's Moon.** Computed from
Eq. 10 against PTM3D's lunar defaults (ks 7.4e-4, kd 3.4e-3, rhos 1100, rhod 1800,
chi 2.7), `k_Martinez / k_Hayne` at the **surface** node (rho = 1100) is:

| T (K) | 30 | 60 | 100 | 150 | 200 | 250 | 300 | 350 | 400 |
|---|---|---|---|---|---|---|---|---|---|
| ratio | 0.134 | 0.279 | 0.457 | 0.611 | 0.682 | 0.693 | 0.676 | 0.651 | 0.628 |

The two laws only cross near rho ~ 1300-1400 kg/m^3. At rho_s they differ by ~35%
**at noon as well as at night**. This is the paper's own behaviour — its Figure 3
shows "New 1100" below "Hayne 1100" across 0-400 K, and my Eq.-10 value at
(400 K, 1100) = 2.34e-3 matches the ~2.2e-3 read off that figure. Surface thermal
inertia falls from 21.9 to 14.8 at 100 K and from 29.9 to 24.9 at 250 K.

So any Haworth delta will contain **both** the intended low-T physics **and** a ~35%
daytime surface-conductivity reduction. It must not be reported as "the low-T fix".

Figures: `docs/figures/conductivity/fig_conductivity_models.{png,pdf}`,
`fig_conductivity_ratio.{png,pdf}`, `conductivity_models.json`; generator
`scripts/plot_conductivity_models.py` (computes the curves in Python directly from
the published coefficients, independently of the C++ — so agreement between the two
is evidence, not tautology).

---

## 2. Implementation

Branch `feat/martinez-conductivity`, worktree `/Users/paha3326/Projects/ptm-martinez`,
based on `b9f3a4b` — deliberately the **exact** commit the PSR overnight campaign's
`R0_coupled_summer` ran on, so a Martinez-vs-R0 comparison is single-variable.

Opt in with:

```yaml
thermophysical:
  conductivity_model: martinez2021   # default: hayne2017 (unchanged)
```

Two design decisions worth preserving:

* **The selector rides `thermal::Config`, never `thermal::Planet`.** `Config` is built
  once in `main` and *passed* to every solver entry point; `Planet` is *rebuilt* from
  the body name at each entry point and therefore discards anything the config put on
  the main-scope Planet. That is the exact mechanism of `BUGS.md` PTM-53, which makes
  `ks_override`/`kd_override` silent no-ops on the Moon/Europa mesh+GPU path. Putting
  the selector on `Config` sidesteps that class of defect by construction — and
  engagement was still proven by byte-difference, not assumed.
* **`setMaterialParams` takes `cond_model` with no default argument**, so a future GPU
  entry point cannot silently run Hayne while the config asked for Martinez. The
  compiler asks instead.

The Martinez branch is taken **before** each SIMD block on the CPU, so the Hayne code
is textually and bit-for-bit untouched. On the GPU, `rho` was already a per-node
buffer, so the law needed **zero** new device buffers; `cond_model` is appended as the
**last** kernel argument so the hard-coded `kArgNTimesteps = 20` / `kArgTOffset = 25`
indices stay valid.

---

## 3. Evidence

All numbers below come from **one** binary: `ptm_thermal_sim` md5
`19f89074c08bbbddc62fdc9db1623797`, `ptm_merge_chunks` `f7cb9893841a0042349b9f50f8e48960`,
`libice_table.dylib` `34c767fdfe39f0003d8032cd486bf45f` (unchanged from the campaign
pin — no ice code was touched), `test_bowl_crater` `770a7ebc3f1f67e438f56335476bbe60`,
`bowl_crater_ptm` `f61a204db6c3b81f1016d163b573d3ff`,
`test_conductivity_model` `3d1452d70cf64b4f84e74660eadf7a28`.

### Anchor test — `build/bin/test_conductivity_model`: 47 checks, 0 failures

`k_am(T)` and `k(T,rho)` reproduce 40-digit exact-decimal references (computed in
Python's `decimal` from the paper's coefficients) at **rtol 1e-12**, at 10, 30, 60,
100, 150, 200, 250, 300, 350, 400 K and rho = 1100 / 1300 / 1800. The float32 OpenCL
device mirror agrees with the host to 1.03e-6 worst-case. Hayne dispatch is
bit-identical to the legacy inline expression. Junk model names are rejected.

*This test caught its own author*: the first reference table was transcribed at 7
digits and 6 of 21 checks failed at rtol 1e-6 on my rounding, not on the code.

### Bowl gate — PASS

`test_bowl_crater` 4/4 sub-tests, F_sky error **0.492713%** (the known-expected value,
memory `ingersoll-fsky-accuracy`), reciprocity max residual 7.0e-14% over 1,966,176
pairs, GPU radiosity converged 4/4. `bowl_crater_ptm` 3 d/D cases, mean_bias
0.0000 / -0.0006 / -0.0010%.

### Do-no-harm — BYTE-IDENTICAL (the discriminating test)

`M0_hayne_repro` = `R0_coupled_summer.yaml` **verbatim** (only `output.directory`
differs; verified field-by-field) run on the new binary, compared with the campaign's
`R0_coupled_summer` produced by the pinned binary `c5e9b976`. Full 120-step coupled
Haworth summer window, GPU persistent kernel, Hapke albedo, corrected VF
(600,378,770 nnz), 140,091 faces:

| field | frames | bytes | result |
|---|---|---|---|
| `temperature` | 120 | 67.2 MB | **bit-equal on every frame** |
| `indirect_ir` | 120 | 67.2 MB | **bit-equal on every frame** |
| `indirect_solar` | 120 | 67.2 MB | **bit-equal on every frame** |
| merged `statistics/*.bin` | 57 files | — | **57/57 md5-identical** |

201.7 MB compared. Independent corroboration: `verify_run` reports T range
24.042612–329.045929 K with 133,746 distinct values, identical to the campaign's
recorded R0 figures.

### Engagement — BYTE-DIFFERENT on every production path

| path | comparison | result |
|---|---|---|
| GPU persistent kernel (24-step radoff, config differs in ONE field) | hayne vs martinez | **24/24 frames differ, max \|ΔT\| = 9.358 K** |
| CPU SIMD batch (12-step ico4) | hayne vs martinez | differ, max \|ΔT\| = 7.839 K |
| CPU scalar `--no-simd-batch` | hayne vs martinez | differ, max \|ΔT\| = 7.530 K |

For reference, the two CPU solvers differ from **each other** by 3.289 K on the
UNCHANGED Hayne law, so that gap is pre-existing and not introduced here.

The engagement test is not ceremony: `ks_override` produces a byte-**identical** field
while printing an override banner (PTM-53). Byte-difference is the only acceptable
evidence that a knob does anything.

### Guards

* Unknown `conductivity_model` → stderr ERROR + FATAL, **exit 1**. Verified.
* Density guard (Eq. 10 gives negative k below rho ≈ 1003.5): unit-tested (rejects
  900 and 1000, accepts 1100). **Its integration path is NOT reachable from a lunar
  YAML** because `thermophysical.rhos` is never read (PTM-67, below) — stated, not
  claimed as tested.

### Short radoff A/B, deep-PSR-floor faces (n = 6248, face set fixed by R0)

martinez − hayne: `deep_mean` **−0.303 K**, `deep_max` **+0.504 K**, `psr_mean`
−0.583 K, `psr_max` +0.554 K, `sunlit_swing` **+1.682 K**.

Direction check passes — lower k → lower thermal inertia → larger diurnal swing.
**This is not the physics answer.** 24 steps with radiation OFF: the deep floor has
not re-equilibrated to the new conductivity, and the wall→floor IR amplification that
dominates the coupled case (worth +25 K in the R0−R1 ladder) is absent by construction.

---

## 4. Latent bugs found and filed (`BUGS.md`)

| id | one line |
|---|---|
| **PTM-62** | HIP/CUDA backend computes `k = kc + R350*T^3` (additive) where every other path uses `kc*(1 + R350*T^3)` — ~700× too large at noon. Dead on macOS/OpenCL, **live on any HIP/CUDA host**. |
| **PTM-63** | `thermal_model.cpp:853`'s comment documents that same wrong formula (code is correct). Probable origin of PTM-62. |
| **PTM-64** | A bare `cmake -DCMAKE_BUILD_TYPE=Release` silently builds a CPU-only binary; configure prints nothing, and the failure only surfaces ~40 s into a run, after which the chunked runner burns two pointless retries. Cost me one aborted run. |
| **PTM-65** | `std::cout` carries `std::fixed << setprecision(1)` forward, so any number printed after ~line 10450 is rendered at one decimal. My own new banner printed `ks (0.0)` for 7.4e-4 and `H=0.1` for 0.06 before I caught it. |
| **PTM-66** | **A malformed YAML config is downgraded to a warning and the entire simulation runs on built-in defaults, exiting 0.** Demonstrated with a config whose only defect was a bad `conductivity_model` value: full run, default output dir, exit 0. This is why the new field is validated in `main`, not by throwing from the parser. |
| **PTM-67** | **Most of the `thermophysical:` block is never read**: `ks, kd, H, rhos, rhod, cp0, Qb, chi, albedo, emissivity` are parsed nowhere, each sitting next to a `use_default_*` flag implying a branch that does not exist. Demonstrated: `use_default_rhos: false` + `rhos: 900.0` ran at 1100. Strictly worse than PTM-53 — dropped at the parser, silently. |

PTM-65, PTM-66 and PTM-67 were each found by *my own* code failing to do what its
banner said, which is the same archetype the ledger exists for.

---

## 5. Flags for Fermi / Sagan — why step 4 was not run

The paper's **formulation is not ambiguous**: Eqs. 9 and 10 are explicit, the
coefficients are tabulated, and my implementation reproduces both to 1e-12 and matches
Figure 3 by inspection. The open questions are about **port scope**, and they change
how a Haworth number should be read:

1. **The change is not confined to low temperature.** ~35% lower surface k at *all*
   temperatures (table in §1). A "Martinez vs Hayne" Haworth delta is a
   conductivity-law delta, not a "low-T correction" delta. Is that the comparison
   Paul wants, or does he want the low-T part isolated?
2. **The paper's retuned `H = 5.6 cm` is not applied, and cannot be** — PTM-67 means
   there is no config route to `H` on a named body. The paper fitted A(rho)/B(rho)
   *and then* retuned H to recover the equatorial diurnal curve. Running with H = 6 cm
   isolates the conductivity law but is **not** a reproduction of their model. Closing
   PTM-67 (or using a per-face `h_parameter_map`) is a prerequisite for that claim.
3. **`ks`, `kd` and `chi` become inert.** Every lunar config in the repo sets them;
   under Martinez they do nothing for conduction. The banner says so, but any
   published table of "model parameters" alongside a Martinez run must not list them.
4. The Feng et al. (2020) albedo the paper pairs with this law is also not applied
   (PTM3D offers Hayne-angular or Hapke).

My recommendation: run `R_mart` as-is and report it explicitly as *"conductivity law
swapped, everything else held at the campaign's values"* — which is a clean
single-variable result and directly answers Paul's question about the PSR floor — while
stating that it is **not** a reproduction of Martinez et al.'s published model. But
that framing is Sagan's call, and the H question is worth answering first.

---

## 6. How to run step 4 when cleared

```bash
cd /Users/paha3326/projects/psr_overnight
nohup /Users/paha3326/anaconda3/envs/europa_viz/bin/python3 \
      orchestrate_martinez.py --with-production \
      > logs/orchestrator_martinez.log 2>&1 &
```

`runs/R_mart.yaml` already exists and differs from `R0_coupled_summer.yaml` in exactly
two fields (`output.directory`, `thermophysical.conductivity_model`), verified
programmatically. The driver re-verifies the pinned binary md5 before every run and
aborts on mismatch; `M0/M1` are recorded verified in `ledger_martinez.jsonl` and will
be skipped. Then:

```bash
python3 analyze_martinez.py --prod
```

Artifacts: driver `orchestrate_martinez.py`, analysis `analyze_martinez.py`, ledger
`ledger_martinez.jsonl`, results `martinez_analysis.json`, superseded first-round runs
under `_superseded/`.

## 7. What was NOT done

* `R_mart` (step 4) — gated, see §5.
* The Diviner cold-end comparison — depends on `R_mart`.
* PTM-62 not fixed: this branch cannot build or test the HIP/CUDA backend.
* PTM-64/65/66/67 not fixed beyond the narrow scope of this feature; each is a
  file-wide or repo-wide change that does not belong in a conductivity branch.
* No two-band / Europa / asteroid path was exercised. The do-no-harm proof covers the
  **lunar GPU mesh coupled** path only; the CPU proof covers a **Moon ico4 radoff**
  run. Nothing here says anything about Europa or a shaped asteroid.
