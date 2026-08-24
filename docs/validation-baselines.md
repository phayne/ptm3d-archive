# Validation baselines — what section 1 of `ptm-validate` actually consumes

Byte-identical regression is the contract for any change that claims "no behavior
change": a solver, GPU, BVH, radiosity, or mesh-pipeline refactor must reproduce
a known-good result bit for bit. This document defines the known-good result —
which commit, which configs, which arms, which output files, and which
toolchain — so that a byte comparison is answering the question the operator
thinks it is answering.

It also records how the previous baselines became unusable, and the one command
that detects the same failure early next time.

---

## 1. Why the `pre-multi-gpu-*` tags are gone

The suite used to name three tags: `pre-multi-gpu-baseline`,
`pre-multi-gpu-bvh-baseline`, and `pre-multi-gpu-radiosity-baseline` (all
May 2026). On 2026-07-20 `main` was **re-rooted** as a public-release baseline.
Its root commit is `a362ee4` (2026-07-20); the old history's root was `bcd3678`
(2025-07-16). The two share no commit:

```
git merge-base origin/main pre-multi-gpu-baseline     # exit 1, empty output
```

All three tags hang off the discarded history. Checking one out yields a May
engine with no two-band radiosity, no row-norm operator, no equilibration seed,
and none of the PTM-89/98/106 radiation fixes. Byte-identity to it is neither
achievable nor desirable, so section 1 as written was unexecutable — not
failing, *unexecutable*, which is worse, because a suite that cannot run
produces no red flag.

**Detection.** A baseline tag is only usable if it is an ancestor of the branch
you are validating. Check it before you trust a comparison, and check it in CI
the day CI exists:

```
git merge-base --is-ancestor validate-baseline-2026-08-15 origin/main \
  && echo "baseline reachable" \
  || echo "ORPHANED — the baseline is not an ancestor of main; re-cut it"
```

**Recovery, if it ever orphans again.** Do not retag and do not force. Cut a
*new* dated tag on current `main`, re-bank the arms on that commit, update this
document and the `ptm-validate` skill to name it, and leave the orphaned tag in
place as history. Superseding beats deleting: an orphaned tag still tells you
what the engine looked like when someone last cared.

---

## 2. The current baselines

Both tags point at **`ecf635a`** (`Merge pull request #95 from
phayne/perf/rad-defaults-and-rho-advisory`), which is `origin/main` as of
2026-08-15 and the first commit carrying the full 2026-08-15 merge set.

| Tag | Blesses | Reproducible from a clean clone? |
|---|---|---|
| `validate-baseline-2026-08-15` | The thirteen **portable** arms: lunar crater (CPU coupled, radiation off, GPU thermal pool at N=1/2/4, GPU radiosity pool at N=1/2/4, gray radiosity at N=1/4), Bennu, Keplerian asteroid, Europa | **Yes** — repository configs plus data shipped in `data/` |
| `validate-baseline-flagship-2026-08-15` | The two **site-local** flagship arms: 509,337-face lunar mesh, N=2 and N=4 GPU thermal | **No** — needs the moon3d mesh and corrected view-factor CSR, which live outside the repository on this machine |

The split is not cosmetic. A colleague with a fresh checkout can run the first
suite and cannot run the second; a suite that pretends otherwise sends them
hunting for a 4 GB file that was never distributed.

**The flagship arms carry a pinned environment, and it is part of the baseline.**
They were run with `PTM_ILLUM_TRUE_NORMALS=1`, `PTM_ILLUM_RISE_HABS=0`,
`PTM_HMATRIX_VF_REPLACE=<corrected CSR>` (the production illumination and
view-factor settings) and `PTM_MAX_TIMESTEPS_PER_DISPATCH=50` (a wedge
mitigation that also fixes the GPU dispatch chunking). Reproducing their bytes
requires the same environment; the exact strings are in the flagship manifest.
⛔ The 25 GB `.vf` is **loaded, never rebuilt** (PTM-19), and then replaced by
the corrected CSR.

Banked digests, per-arm run logs, and full provenance:
`~/projects/validate_rebaseline_2026-08-15/bank/baseline_manifest.json`
(portable) and `.../bank/baseline_manifest_flagship.json` (site-local).

---

## 3. The arms

Every arm is one config plus, at most, command-line overrides. Each was chosen
because it reaches a code path no other arm reaches; an arm that duplicates
another's coverage costs wall-clock time and buys nothing.

| Arm | Config | What only this arm covers |
|---|---|---|
| `moon_crater80N_coupled_cpu` | `examples/tutorial04_crater80N_coupled.yaml` | CPU thermal, H-matrix view factors, CPU two-band radiosity, the coupling loop |
| `moon_crater80N_vfbuild_sparse` | `examples/validation/baseline_crater80N_vfbuild_sparse.yaml` | The SPARSE view-factor builder, matrix BUILT not loaded — the only arm that can see a change to the near-field estimator |
| `moon_crater80N_vfbuild_hmatrix` | `examples/validation/baseline_crater80N_vfbuild_hmatrix.yaml` | The H-MATRIX builder, matrix BUILT not loaded, including its ACA low-rank compression |
| `moon_crater80N_radoff_cpu` | `examples/tutorial04_crater80N_radoff.yaml` | Illumination plus CPU thermal with radiation disabled |
| `moon_crater80N_gputhermal_n{1,2,4}` | `examples/validation/baseline_crater80N_gputhermal.yaml` | GPU persistent-kernel thermal solver, the multi-GPU **thermal** pool, and the two-phase cadence that makes the 2026-08-15 defaults active |
| `moon_crater80N_radgpu_n{1,2,4}` | `examples/validation/baseline_crater80N_radgpu.yaml` | GPU radiosity solver and the multi-GPU **radiosity** pool (needs a sparse view-factor matrix; the GPU solver refuses an H-matrix) |
| `asteroid_bennu_spice_cpu` | `examples/quickstart_asteroid_bennu.yaml` | Generic-body thermophysics on a real SPICE ephemeris and an imported facet shape |
| `asteroid_keplerian_cpu` | `examples/quickstart_asteroid_diurnal.yaml` | Generic-body thermophysics on the analytic Keplerian propagator |
| `europa_diurnal_cpu` | `examples/quickstart_europa_diurnal.yaml` | Europa built-in body, primary-body infrared and reflected light, a full eclipse inside the output window |
| `moon509k_gputhermal_n{2,4}` | site-local (see the flagship manifest) | Production scale: 509,337 faces, corrected CSR view factors, 600 equilibration steps |

**The generic-body arms are not optional.** A passing lunar gate says nothing
about a change gated to the generic-body path — for such a change "Moon
byte-identical" is a tautology, because the gate excluded the Moon by
construction. Commit `5bfbdbe` scrambled the asteroid diurnal field while
passing the bowl-crater test, lunar byte-identity, and the seam check.

### Two arms exist because the other thirteen are blind to the view factors

Added 2026-08-23 on `fix/hmatrix-symmetric-quadrature`. Every other arm in the
suite is **structurally inert** to a change in how view factors are computed:

| arm | why it cannot see a view-factor change |
|---|---|
| `asteroid_bennu_spice_cpu`, `asteroid_keplerian_cpu`, `europa_diurnal_cpu` | `radiation.enabled: false` |
| `moon_crater80N_radoff_cpu` | radiation off |
| `moon_crater80N_coupled_cpu` | LOADS a cached H-matrix `.vf` |
| `moon_crater80N_radgpu_n{1,2,4}`, `moon_crater80N_gputhermal_n{1,2,4}` | LOAD a prebuilt `.vf` |

This is not hypothetical. The 2026-08-23 symmetric near-field quadrature fix
moves sunlit crater-rim faces by up to **+13.45 K**, and came out **5/5
byte-identical** against the CPU arms of this suite. A green suite there was a
tautology, not do-no-harm evidence, and it would have said exactly the same
thing about a fix that was wrong.

The two new arms differ from every other one in a single load-bearing way:
`radiation.view_factor_file: ''`, so the matrix is **built on every run**. One
takes the sparse builder, one the H-matrix builder, because PTM3D has two
in-engine view-factor implementations with two different near-field order rules
and `ptm_thermal_sim` chooses between them by a memory threshold as well as by
config. An arm covering one says nothing about a run that took the other.

Confirm from the log that the matrix was built, not loaded:

```
grep -E 'Using sparse view factor matrix|Building H-matrix' <log>   # must appear
grep -E 'Loading view factor'                              <log>   # must NOT
```

Sensitivity is verified, not assumed. Flipping
`radiation.nearfield_quadrature_order` from 3 to 1 — which makes the near field
a bare M1 centroid estimate — moves **9 of 32 output files** on BOTH arms
(`temperature_avg/min/max`, `indirect_solar_max`, and their `_raw` counterparts);
`ptm_validate_baseline.py equivalence` returns `NOT EQUIVALENT`, exit 1. That
negative control is what makes a green verdict from these arms mean something.

⛔ **These two arms are NOT in `validate-baseline-2026-08-15`, and their bytes
are not comparable to it.** They are new configs, and they run physics that
changed on purpose: the near-field estimator was one-sided until 2026-08-23 and
read −21.7 percent on edge/vertex-touching pairs. Their baselines are banked
separately, at the branch and commit named in their own manifest, with the
attribution in `notes`. Do not merge them into the 2026-08-15 manifest and do
not retag it.

### Two arms exist only to keep a gate honest

The two `examples/validation/` configs are not variants for variety's sake:

- `baseline_crater80N_gputhermal.yaml` sets
  `equilibration_samples_per_diurnal_cycle: 24`, which makes the equilibration
  cadence differ from the output cadence and the run **two-phase**. That is the
  only shape in which the 2026-08-15 `rad-skip-eq` and `rad-select-window`
  defaults are active. On `tutorial04_crater80N_coupled.yaml` both print
  `STOOD DOWN`, so a suite built on the tutorial config alone would bless
  nothing about the defaults it claims to cover.
- `baseline_crater80N_radgpu.yaml` sets `use_hmatrix: false`, because the GPU
  radiosity solver refuses an H-matrix operator and silently falls back to the
  CPU. With an H-matrix, `--rad-gpu-devices 4` runs exactly the same CPU code as
  `--rad-gpu-devices 1`, and the resulting "byte-identical" verdict is a
  fallback tautology rather than evidence about the pool.

Before trusting either arm, confirm from its log that the path engaged:

```
grep -E '\[rad-skip-eq\]|\[rad-select-window\]' <gputhermal log>   # must NOT say STOOD DOWN
grep -E '\[Radiosity\] ACTIVE RUNG|Forcing the CPU' <radgpu log>   # must name the GPU rung
```

---

## 4. What is compared

Everything under a run's `statistics/` directory, **recursively** — including
the `_raw/` subdirectory written by `output.write_raw_stats`. The historical
six-file comparison (temperature and illumination, average/minimum/maximum) is
a subset: the current suite compares **16 to 32 files per arm** depending on
which channels the arm produces — 16 for the radiation-off and generic-body
arms, 22 for the coupled CPU crater, 26 for the gray arms, 32 for the arms that
also write raw statistics. 330 files across the 13 portable arms.

Excluded, by name, because their bytes legitimately vary between runs:
`provenance.txt` (records wall-clock time and the invoking command),
`manifest.json`, `.DS_Store`.

Two traps worth stating outright:

- `statistics/*.bin` begins with a `uint32` header, so a naive floating-point
  comparator can report "identical" on files that differ. **Compare bytes.**
- `effective_albedo.bin` legitimately carries `NaN`, so a naive comparator
  reports "differs" at a maximum absolute difference of zero. Again: compare
  bytes, not values.

---

## 5. Running the suite

`scripts/ptm_validate_baseline.py` does the digesting and comparing. It never
runs the simulator, so it is safe on a machine with no GPU.

```
# Compare two finished run directories directly (N=1 versus N=4, CPU versus GPU)
python3 scripts/ptm_validate_baseline.py equivalence <run_a> <run_b>

# Compare a set of finished runs against the banked baseline
python3 scripts/ptm_validate_baseline.py verify \
    --manifest ~/projects/validate_rebaseline_2026-08-15/bank/baseline_manifest.json \
    --repo . --binary build/bin/ptm_thermal_sim
```

`verify` reports **provenance before bytes**. If a config file's bytes have
changed since the baseline was cut, if the engine commit differs, or if a
linked dynamic library differs, it says so first — because a byte comparison
across any of those changes answers a different question than the one you
asked, and the byte verdict alone would mislead you.

⚠️ **Until the branch that adds this document is merged, a bare checkout of
`ecf635a` does not contain `examples/validation/`.** Running `verify --repo`
against such a checkout correctly reports those three configs as `GONE` and
returns a MISMATCH *even though all 330 files are byte-identical* — the tool is
refusing to certify a comparison whose reference configs it cannot see. That is
working as designed, not a false alarm. Verify from a checkout that has the
configs.

The engine commit is read from each run's `provenance.txt`, not from the
repository's `HEAD`: the sha is baked in at build time, so a manifest banked
after an unrelated commit would otherwise credit the arms to an engine that
never ran them. If two arms report different engine shas, `bank` refuses
outright — a mixed manifest blesses nothing.

---

## 6. What the baselines were built with, and what invalidates them

### Embree

Byte-identity is a property of the **whole** toolchain, not of the engine
binary. `ptm_thermal_sim` links Embree dynamically, so an Embree swap moves
results with the engine binary's checksum completely unchanged.

These baselines were built against **Homebrew Embree 4.3.3, `EMBREE_MAX_ISA=SSE2`**:

| | |
|---|---|
| Library | `/usr/local/opt/embree/lib/libembree4.4.dylib` |
| Resolves to | `/usr/local/Cellar/embree/4.3.3/lib/libembree4.4.dylib` |
| md5 | `e0b68a29586192406d34f4213b7c505c` |
| Size | 9,908,824 bytes |
| Kernels | SSE2 only |

A second Embree of the same version, built multi-ISA, is installed at
`~/opt/embree-4.3.3-avx512` (md5 `7e229da52bb90938741bba6d8666f3c7`, 47,567,080
bytes, kernels SSE2 SSE4.2 AVX AVX2 AVX512). Selecting it with
`-DEMBREE_ROOT=$HOME/opt/embree-4.3.3-avx512` **changes lunar bytes**: measured
2026-08-15, 75 of 260 files differ on a lunar site run, worst face-frame
difference **0.00907 K**. Europa and Bennu are byte-identical across the swap.

So: **a comparison run under a different Embree is not a section 1 comparison.**
Either build both arms against the same Embree, or state the ISA and treat
lunar differences at the 0.01 K scale as toolchain, not regression. Every run
records `embree_library` and `embree_max_isa` in its `provenance.txt`, and the
configure step prints the ISA; read them rather than assuming.

`test_bowl_crater` cannot referee this. It builds its view factors **without
Embree** (`test_bowl_crater.cpp`), so passing it is a do-no-harm check, never a
discriminating test of an Embree change. The Embree-exercising bowl target is
`bowl_crater_ptm`.

### The 2026-08-15 defaults are baked in

These baselines were cut *after* the radiation-gate defaults flipped. Three
behaviors are therefore **in** the banked bytes:

| Default | Kill switch |
|---|---|
| Equilibration-step radiosity skip, on | `PTM_RAD_SKIP_EQ_STEPS=0` |
| Output-window face selector, on | `PTM_RAD_SELECT_OUTPUT_WINDOW=0` |
| Picard coupling cap raised from 10 to 16 | `radiation.max_coupling_iterations: 10` |

Both environment variables are **tri-state**: unset means stand down and print
why, `=1` means require the path and fail loudly if it is unavailable, `=0`
means off. Only `1/true/on/yes` and `0/false/off/no` parse; anything else is
fatal at configuration-validation time.

**Comparing across the default flip requires the kill switches on both arms.**
Without them, the flip's own intended change (measured at the time: 6 of 22
files, 33 of 509,337 faces, at most 2.5e-5 K) is indistinguishable from a
regression.

### Other things that invalidate a comparison

- **Rebuilding between arms.** Build once and run both arms with that binary, or
  rebuild both arms identically.
- **A changed reference config.** `verify` checks each config's md5 against the
  banked value and fails loudly. A config edit does not make the baseline wrong;
  it makes the baseline *about something else*.
- **A different machine.** The GPU arms are pinned to this host's four AMD
  W6800X Duo compute devices. Nothing here claims cross-machine byte-identity.
- **A different thread count.** Several OpenMP reductions in the solver and in
  the statistics accumulators sum in thread-partition order, so the same binary
  on the same config at 8 and at 48 threads writes different bytes. Measured
  2026-08-23 on `moon_crater80N_vfbuild_sparse`: at 48 threads 32/32 files match
  the `vfunify-stage1-2026-08-23` manifest; at 8 threads, 31/32 — one file,
  `statistics/_raw/temperature_sum.bin`, and nothing else changed. Since that
  date the engine records `omp_max_threads` in `provenance.txt`, `bank` copies
  it into the manifest, and `verify` reports a thread-count change **before**
  the byte verdict. An arm banked by an older engine records no thread count and
  `verify` says so; treat its byte verdict as unattributable until it is re-cut.

---

## 7. Equivalences validated at `ecf635a`

Measured on 2026-08-15 with the binaries and Embree recorded above. Every GPU
arm's log was checked to confirm the intended dispatch rung was actually active,
so none of these is a fallback tautology.

| Equivalence | Result |
|---|---|
| Same binary, same config, run twice (CPU crater) | 22/22 files byte-identical — the engine is self-reproducible, which is the precondition for everything else here |
| GPU thermal pool, N=1 versus N=2 | 32/32 byte-identical |
| GPU thermal pool, N=1 versus N=4 | 32/32 byte-identical |
| GPU radiosity pool, N=1 versus N=2 | 32/32 byte-identical |
| GPU radiosity pool, N=1 versus N=4 | 32/32 byte-identical |
| Gray radiosity, N=1 versus N=4 | 26/26 byte-identical (see the caveat below) |
| Flagship 509k GPU thermal, N=1 versus N=2 versus N=4 | 32/32 byte-identical, twice independently (see section 9) |

`gpu_devices: 4` is therefore evidence-backed on this machine for both pools.
**No default was changed** — the shipped default remains `1`. Whether production
should move to 2 or 4 is a separate decision.

### A known mode difference, measured and found null here

In **gray** (non-band-separated) radiosity the single-GPU path and the pool
configure the solver differently: the single-GPU path clears `warm_start` only
when two-band is active, while the pool clears it unconditionally
(`radiation_exchange.cpp`, "Cross-batch warm-start has no meaning across
devices"). A warm-started and a cold-started iteration can stop at different
iterates, so N=1 and N>1 may legitimately differ in gray mode.

Measured on the 12,800-face crater in gray mode, N=1 versus N=4: **26/26
byte-identical**. The hazard is real in the code and did not materialize here.
The crater's view-factor spectral radius is 0.134, so its iteration converges
almost immediately from either start; a high-spectral-radius mesh (the 509k
flagship is 0.776) is where this would show up, and no gray-mode arm exists at
that scale. **Treat a gray-mode N=1-versus-N>1 difference as a known mode
difference to report, not as a suite failure** — and not as evidence that
two-band is affected, because two-band clears `warm_start` on both paths.

---

## 9. Flagship scale: the N=1/2/4 ladder and what it cost

Measured on the 509,337-face lunar mesh with the corrected view-factor CSR
loaded (nnz 495,163,704; ⛔ never rebuilt, PTM-19), 600 equilibration steps and
20 output steps, `gpu_batch_size: 4000`, `PTM_MAX_TIMESTEPS_PER_DISPATCH=50`.

| Arm | Wall clock | GPU kernel device-time | Kernel dispatches |
|---|---|---|---|
| N=1 | 1834 s | 1588.6 s | 687 |
| N=2 | 1112 s | 1586.2 s | 687 |
| N=4 | 650 s | 1588.9 s | 687 |

**Bytes: N=1 == N=2 == N=4, 32/32 identical.** The N=2-versus-N=4 result was
obtained **twice independently** — once on a contended pair and once on the
clean pair — and byte-identity is not a timing property, so contention cannot
manufacture it.

Measured end-to-end speedups: **1.649× at N=2, 2.822× at N=4** (both against
N=1), and **1.711× for N=4 over N=2**.

Two things make the speedup credible rather than merely observed:

- **The device-time is the same on all three arms** (1588.6 / 1586.2 / 1588.9 s,
  a spread of 0.17%) and so is the dispatch count (687). Every arm did the same
  arithmetic; the gain is parallel execution, not less work.
- **The wall difference is attributable.** Every other phase is device-count
  independent, and the one other phase the log times directly — the coupling's
  indirect-flux sweeps — is 50 s at N=2 and 51 s at N=4.

### ⚠️ The third point is what made the scaling honest

There is no per-phase timer in the log, so isolating the thermal phase requires
a model: `T(N) = F + C/N`, with `F` device-count independent. **Two points
determine that model exactly, and the answer you get depends on which two:**

| Points used | `C` (parallelizable) | `F` (fixed) |
|---|---|---|
| N=2 and N=4 | 1848 s | 188 s |
| N=1 and N=2 | 1444 s | 390 s |

They disagree by 28%, so `1/N` is a good approximation and **not exact**. A
least-squares fit over all three points gives `C` = 1559 s and `F` = 289 s — a
parallel fraction of 0.850 — with residuals of −0.8%, +3.9% and −4.4%.

Therefore: **the 1.711× for N=4 over N=2 is measured; a thermal-phase-only
speedup near 2× is model output, not a measurement.** Quote the first, and if
you quote the second, say it is derived and name the residuals. Had this been
stopped at two arms, the report would have carried `C` = 1848 s as a fact.

⚠️ **Contention is the dominant error source in any of these numbers.** The same
N=2 arm measured 1297 s while an unrelated job held the GPU and 1112 s alone: a
14% penalty, consistent with the 18.5% previously documented on this host for
identical code paths. Never quote a speedup taken beside another job.

## 8. What this suite does not cover

Stated explicitly, because the value of a gate is bounded by what it touches:

- **Scientific accuracy.** Byte-identity is a *refactor* contract. It says the
  numbers did not move; it says nothing about whether they were right. Diviner
  cross-validation (section 4 of the `ptm-validate` skill) is the data-anchored
  check, and the Ingersoll bowl-crater test is the analytic one.
  Concretely: the flagship arms use a deliberately short shape (600
  equilibration steps, 20 output steps) built for cost and equivalence probing,
  and `scripts/ptm_qa.py thermal` **fails** its never-lit-face coherence check
  on them — on this baseline's run (standard deviation of the minimum
  temperature, 21.65 K against a 3.0 K threshold) *and* on the earlier
  independent run of the same recipe at `d279eca` (13.95 K). The failure is a
  property of the recipe, not of the baseline. **Never quote a flagship arm as a
  converged thermal field.**
- **The exosphere, camera, and merge-chunks binaries.** Only `ptm_thermal_sim`
  is exercised here.
- **Illumination on the GPU.** Every arm traces shadow rays on the CPU with
  Embree. `--use-gpu-illumination` is untested by this suite.
- **Chunked runs and warm start.** The `initial_state_file` /
  `final_state_file` path and `ptm_merge_chunks` have their own equivalence
  requirement and no arm here.
- **Cross-machine reproducibility**, as above.
- **Gray-mode radiosity at production spectral radius**, as above.
- **Any Europa or asteroid GPU path.** The generic-body and Europa arms are all
  CPU; a GPU change gated to those bodies would pass this suite untested.
