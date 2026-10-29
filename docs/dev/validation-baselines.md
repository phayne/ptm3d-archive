# Validation baselines: what `scripts/ptm_validate_baseline.py` consumes

Byte-identical regression is the contract for any change that claims "no behavior
change": a solver, GPU, BVH, radiosity, or mesh-pipeline refactor must reproduce
a known-good result bit for bit. This document defines the known-good result
(which commit, which configs, which arms, which output files, and which
toolchain) so that a byte comparison is answering the question the operator
thinks it is answering.

It also records how the previous baselines became unusable, and the one command
that detects the same failure early next time.

---

## 1. Retirement of the `pre-multi-gpu-*` tags

The suite used to name three tags: `pre-multi-gpu-baseline`,
`pre-multi-gpu-bvh-baseline`, and `pre-multi-gpu-radiosity-baseline` (all
May 2026). On 2026-07-20 `main` was re-rooted as a public-release baseline.
Its root commit is `a362ee4` (2026-07-20); the old history's root was `bcd3678`
(2025-07-16). The two share no commit:

```
git merge-base origin/main pre-multi-gpu-baseline     # exit 1, empty output
```

All three tags hang off the discarded history. Checking one out yields a May
engine with no two-band radiosity, no row-norm operator, no equilibration seed,
and none of the 2026-08 radiation fixes (chunk-independent coupling, exchange selection, source albedo). Byte-identity to it is neither
achievable nor desirable, so section 1 as written was unexecutable: not
failing, *unexecutable*, which is worse, because a suite that cannot run
produces no red flag.

Detection. A baseline tag is only usable if it is an ancestor of the branch
you are validating. Check it before you trust a comparison, and check it in CI
the day CI exists:

```
git merge-base --is-ancestor validate-baseline-2026-08-15 origin/main \
  && echo "baseline reachable" \
  || echo "ORPHANED — the baseline is not an ancestor of main; re-cut it"
```

Recovery, if it ever orphans again: do not retag and do not force. Cut a
*new* dated tag on current `main`, re-bank the arms on that commit, update this
document to name it, and leave the orphaned tag in
place as history. Superseding beats deleting: an orphaned tag still tells you
what the engine looked like when someone last cared.

---

## 2. The current baselines

### 2.1 The current baseline: `postmerge-baseline-2026-09-05` at `c137564`

Location (deliberately protected):

```
~/projects/ptm-validation-bank-2026-09-05/
    baseline_manifest_2026-09-05.json     the manifest (schema 3)
    runs/<arm>/statistics/                the banked output BYTES, 394 files
    inputs/crater80N_{coupled,hmat,sparse}.vf   the view-factor matrices the arms load
    DO-NOT-DELETE.md                      what this directory is, and why it must not be swept
```

Never delete this directory without the maintainer's approval. Its predecessor
(`postmerge-baseline-2026-08-24`, at `<workdir>/postmerge_2026-08-24/`)
was removed in a routine disk cleanup on 2026-09-03 because nothing on disk
said it was needed, taking the corrected `crater80N_sparse.vf` with it and
leaving the project with no executable regression gate at all. Two things now
stand against a repeat: the `DO-NOT-DELETE.md` above, and
`ptm_validate_baseline.py`, which refuses to write anywhere beneath a
directory carrying that marker.

Cut on `origin/main` `c137564` (the merge of `feat/state-depth-starter`,
pull request #171). It blesses the same fifteen arms as its predecessor
(the five portable CPU arms, the two `vfbuild` arms, and the eight GPU arms),
394 files, all on one binary:

| | |
|---|---|
| Engine commit | `c13756483aed9c65198e7407dec03881e2daa872`, read from each arm's own `provenance.txt` |
| `ptm_thermal_sim` md5 | `90c7c8ebc557af30e6a0a24ed0c08270` |
| Build | the canonical `./rebuild.sh` configuration: `PTM_ENABLE_GPU=ON`, `GPU_BACKEND=OPENCL`, `PTM_ARCH_NATIVE=ON`, `BUILD_SHARED_LIBS=ON`, `CMAKE_BUILD_TYPE=Release` |
| Embree | `~/opt/embree-4.3.3-avx512`, md5 `7e229da52bb90938741bba6d8666f3c7`, max ISA AVX512 |
| Threads | `OMP_NUM_THREADS=48`, recorded per arm |
| Bowl-crater gate at this commit | `PASS: 16 of 16 verdicts (0 skipped, 4 latitudes)`; analytic `F_sky` error 0.492713% against a 1% tolerance, the same value recorded at `ecf635a` |

This bank is not byte-comparable to the 2026-08-15 or 2026-08-24 banks on
any lunar arm, and the reason is the toolchain, not the physics. Those were
cut against Homebrew's SSE2 Embree (md5 `e0b68a29…`); `rebuild.sh` has
owned the build configuration since 2026-09-02 and selects the widest-ISA
Embree it can find, which on this host is the multi-ISA AVX512 build. Section 6
measures that swap at 75 of 260 files and 0.00907 K on a lunar site run. This
bank records what an operator on this machine *actually gets today*; comparing
it to an older bank answers a different question, and `verify` says so before
it prints any byte verdict.

This bank covers less than its predecessor. `postmerge-baseline-2026-08-24` carried three manifests (`PTM_ARCH_NATIVE`
`OFF`, `ON`, and a separate GPU build). This one has a single configuration. So
there is no ISA twin and no CPU-only-build twin on `c137564`: a byte
difference read from a `--portable` or `--cpu-only` build against this bank is
unattributable until such a twin is cut. `verify` reports the build-flag
difference prominently rather than letting it pass, but reporting it is not
covering it. Cutting the twin is one build and one command:

```
./rebuild.sh --portable                     # into build/, PTM_ARCH_NATIVE=OFF
python3 scripts/ptm_validate_baseline.py run  --repo . --binary build/bin/ptm_thermal_sim \
    --run-root <scratch>/runs --inputs-dir ~/projects/ptm-validation-bank-2026-09-05/inputs
python3 scripts/ptm_validate_baseline.py bank --repo . --binary build/bin/ptm_thermal_sim \
    --tag postmerge-baseline-2026-09-05-portable \
    --bank-dir ~/projects/ptm-validation-bank-2026-09-05-portable \
    --out baseline_manifest_portable.json --arm <name>=<rundir> ...
```

Prerequisite: the SPICE kernels. Three kinds of arm (`asteroid_bennu_spice_cpu`,
`europa_diurnal_cpu`, and every lunar arm) need `data/kernels/{lsk,pck,spk,fk}`,
which is gitignored, 1.2 GB, and therefore absent from every fresh worktree.
Without it each arm dies on `FATAL: SPICE initialization failed for body
'Moon'`. `run` and `verify --rerun` check for this before they burn a run and
name the fix. Link them from a checkout that has them rather than
re-downloading per worktree:

```
python3 scripts/fetch_kernels.py --link-from ~/Projects/ptm
```

That checks every pinned sha256 before it links, labels the two files with
no pinned hash (`spk/jup365.bsp`, and the `pck/pck00010.tpc` + `pck/gm_de431.tpc`
extras) as `UNPINNED` rather than as verified, and refuses to overwrite a real
file with a link.

The bank is self-contained and read-only. It holds the output bytes, not a
path to them, and it holds the three view-factor matrices the arms load
(`crater80N_coupled.vf` and `crater80N_hmat.vf` are byte-identical, md5
`7e93d2a16b1e7cf9c8a825f26dcf64ea`, 36,341,754 bytes; `crater80N_sparse.vf` md5
`8e5fbe76247e42d63f7189e4ba23784d`, 15,621,908 bytes). Its manifest is schema
`ptm3d-validation-baseline/3`, which additionally records each arm's recipe
(the config and the command-line overrides), so `verify --rerun` can replay
the suite out of the bank itself rather than out of a script somebody has to
still have.

#### Round-trip evidence for this bank, 2026-09-05

| Check | Result |
|---|---|
| The 15 arms, run and banked | 394 files; every arm's log confirmed the intended dispatch rung (`Multi-GPU dispatch across 2/4 device(s)`, `ACTIVE RUNG: single-GPU` versus `multi-GPU pool`, `[rad-skip-eq]`/`[rad-select-window]` live, the two `vfbuild` arms building, not loading), checked by the runner, not by eye |
| A fresh copy of the bank verified against the manifest | 394/394 byte-identical, 15 arms, `VERDICT: BYTE-IDENTICAL`, exit 0 |
| A copy of the bank with one bit flipped in `moon_crater80N_gputhermal_n4/statistics/temperature_max.bin` (same file length) | 393/394, `DIFFERS moon_crater80N_gputhermal_n4: 31/32`, names the file, `VERDICT: MISMATCH`, exit 1; the other 14 arms still `IDENTICAL` |
| `verify` with no `--arm` and no `--rerun` | exit 2, no verdict printed at all (section 5) |
| `verify --arm <name>=<the banked directory>` | exit 2, refused as a self-comparison |

The one-flipped-bit arm is the essential one. A gate that has never been
observed to fail is not known to be a gate.

The predecessor bank, `postmerge-baseline-2026-08-24` at `51bf168`, is gone
from disk (the whole `<workdir>/postmerge_2026-08-24/` directory,
including the manifests below), which are recorded here only so that a reference
to them is recognizable as a reference to something that no longer exists:

```
<workdir>/postmerge_2026-08-24/bank/baseline_manifest_postmerge_cpu.json                GONE
<workdir>/postmerge_2026-08-24/bank/baseline_manifest_postmerge_cpu_archnative.json        GONE
<workdir>/postmerge_2026-08-24/bank/baseline_manifest_postmerge_gpu.json                GONE
```

The rest of this section records what that bank measured, which remains the
attribution history for the arms even though its bytes are unrecoverable.

One arm of that bank had been superseded on its own, by a data change
rather than a code change:

```
~/projects/bennu_spk_2026-08-27/bank/baseline_manifest_bennu_sb118_2026-08-27.json
~/projects/bennu_spk_2026-08-27/bank/baseline_manifest_bennu_sb118_archnative_2026-08-27.json
```

`asteroid_bennu_spice_cpu`, 2026-08-27, re-banked for the ephemeris upgrade.
Bennu's shipped kernel moved from JPL solution 76 (2013; `data/horizons_cache/
bodies/101955_Bennu_(1999_RQ36)/ephemeris.spk`, coverage 2015-01-01 to
2023-05-31) to solution 118 long-span (`data/kernels/spk/
sb-101955-118_long.bsp`, fetched, coverage 1900-01-01 to 2135-10-01,
`https://ssd.jpl.nasa.gov/ftp/xfr/sb-101955-118_long.bsp`, sha256
`2e25094e…a5c3b48f`), so that a present-day start date is inside coverage
instead of silently reaching the Keplerian fallback. Both re-banks ran on the
same two binaries as the manifests they supersede (`PTM_ARCH_NATIVE=OFF`
md5 `9d657590009ade0c0d8f2d07d2faa7a2`, `ON` md5
`59c9f743efe25c26ae0660f566d5173b`, both `51bf1688b0`), so the movement is the
kernel and nothing else: 7 of 16 files, at most 2.14e-4 K on
`temperature_max` and 2.44e-4 W/m² on `direct_solar_max`, one to a few float32
ULP, the same magnitude in both ISA settings; `primary_ir`,
`primary_reflected` and `effective_albedo` are exactly zero. The two kernels
differ by at most 31.97 km over their common 2015–2023 window (`spkdiff`, 1000
samples). Both old manifests are untouched and their other six arms each
remain current; the `_gpu` manifest does not carry this arm at all, so it
needed no re-bank. Before each re-bank the old kernel was re-run on the same
binary and reproduced the superseded digests 16/16, so these numbers are the
kernel's and not the harness's.

The CPU arms are banked twice, because `PTM_ARCH_NATIVE` is part of the
baseline and nobody had recorded it (section 6). The first manifest is the
documented onboarding and CI configuration, `PTM_ARCH_NATIVE=OFF`; the second is
`ON`, the setting the 2026-08-15 builds actually used. They differ from each
other in five files out of 150: the float64 `statistics/_raw/*_sum.bin`
accumulators of the two `vfbuild` arms, at a maximum relative difference of
1.6e-12. The GPU manifest is `PTM_ENABLE_GPU=ON`, `GPU_BACKEND=OPENCL`,
`PTM_ARCH_NATIVE=OFF`, the post-#118 documented onboarding combination, whose
configure step prints `Eigen ABI uniformity: OK`. Every arm ran at
`OMP_NUM_THREADS=48`, recorded per arm, and the build flags are now recorded per
binary in the manifest itself rather than in prose.

No git tag has been cut. The name above is proposed; the manifests bless
the commit. Cutting the tag is a separate, deliberate act.

Which arms still match `validate-baseline-2026-08-15`, and why the others do not:

| Arm | Against `validate-baseline-2026-08-15` | Cause |
|---|---|---|
| `asteroid_bennu_spice_cpu` | 16/16 identical | — |
| `asteroid_keplerian_cpu` | 16/16 identical | — |
| `moon_crater80N_coupled_cpu` | 8/22 | pull request #102 |
| `moon_crater80N_radoff_cpu` | 5/16 | pull request #102 |
| `europa_diurnal_cpu` | 7/16 | pull request #102 |
| `moon_crater80N_vfbuild_sparse` | not in that baseline | new config, section 3 |
| `moon_crater80N_vfbuild_hmatrix` | not in that baseline | new config, section 3 |
| `moon_crater80N_gputhermal_n{1,2,4}` | not comparable | rebuilt view factors, below |
| `moon_crater80N_radgpu_n{1,2,4}` | not comparable | rebuilt view factors, below |
| `moon_crater80N_radgpu_gray_n{1,4}` | not comparable | rebuilt view factors, below |

The three that moved were moved by `45eefe4` (pull request #102,
`perf/primary-disk-face-budget`), and by nothing else. Bisected 2026-08-24 at a
fixed ISA setting, which is the only way this comparison means anything:
`ecf635a` and `a6b141a` (#101) are byte-identical on the radiation-off arm, and
`45eefe4` is byte-identical to `51bf168` on the radiation-off (16/16), coupled
(22/22) and Europa (16/16) arms. Everything merged after #102, including the
entire view-factor arc, moves zero bytes on all five portable CPU arms. The
magnitude on the radiation-off arm is 11 of 16 files, up to 573 of 12,800 faces,
maximum 0.02846 K on `temperature_max`, confined to the illumination,
primary-infrared and primary-reflected channels.

This supersedes the earlier working note that the drift was "pre-existing,
`ecf635a` to `cb1e55d`, cause unidentified". It was pre-existing; the cause is
now identified.

### The GPU arms were re-run on rebuilt view factors, not re-verified

The GPU arms of `validate-baseline-2026-08-15` were not merely stale; they
were unreproducible, so nothing in this baseline is a comparison against them.
The five radiosity arms loaded a `crater80N_sparse.vf` built by the engine sparse
builder during the zero-occlusion window, containing no occlusion at all
(section 6). All seven copies of that matrix are quarantined at
`~/projects/quarantine_zero-occlusion-vf_2026-08-24/`.

Both input matrices were therefore rebuilt on `main`, and both live beside
the manifests with a provenance sidecar:

| Matrix | Consumed by | Format | Content |
|---|---|---|---|
| `crater80N_sparse.vf` | `radgpu`, `radgpu_gray` | `VFFM` v2, md5 `b1e3cdc4…` | 1,297,556 non-zeros; 224,412 of 1,521,968 pairs rejected as blocked |
| `crater80N_hmat.vf` | `gputhermal` | `HMAT` v2, md5 `61454478…` | built twice, identical; supersedes the 2026-08-15 file, which predates the near-field quadrature port |

The H-matrix cache was never contaminated (that builder always recomputed its
own scene shift), but it was replaced anyway, because it carried the pre-#122
near-field physics. The `HMAT` format has no physics provenance stamp: the
old file and the new one both report `version 2`, so a stale H-matrix cache loads
silently. Only the sparse `VFFM` format announces v1 against v2. Check the md5.

Two properties were proven rather than inherited from the CPU work:

- Run-to-run determinism on the GPU. Every one of the eight arms was run
  twice; all eight pairs are byte-identical (32/32 on the six two-band arms,
  26/26 on the two gray arms). The deterministic H-matrix matvec landed on the
  CPU side of the arc; a CPU proof is not a GPU proof.
- `N=1` equals `N=2` equals `N=4` on both pools: `gputhermal` 32/32 for all
  three pairings, `radgpu` 32/32 for all three, `radgpu_gray` 26/26 at N=1
  against N=4.

Neither is a fallback tautology, and that was checked per arm rather than
assumed: `gputhermal` at N=2 and N=4 prints `Multi-GPU dispatch across 2/4
device(s)` and enumerates the devices while N=1 does not, and the radiosity arms
print `[Radiosity] ACTIVE RUNG: single-GPU radiosity` at N=1 against
`multi-GPU pool, 2/4 device(s)` at N=2 and N=4. The radiosity logs also report
`nnz = 1297556`, which is the corrected matrix and not the quarantined one.

The gray-mode hazard in section 7 (the single-GPU path clears `warm_start` only
in two-band mode while the pool clears it unconditionally) again did not
materialize. It remains real in the code, and there is still no gray-mode arm at
production spectral radius.

### 2.2 The 2026-08-15 tags

Both tags point at `ecf635a` (`Merge pull request #95 from
phayne/perf/rad-defaults-and-rho-advisory`), which is `origin/main` as of
2026-08-15 and the first commit carrying the full 2026-08-15 merge set.

| Tag | Blesses | Reproducible from a clean clone? |
|---|---|---|
| `validate-baseline-2026-08-15` | The thirteen portable arms: lunar crater (CPU coupled, radiation off, GPU thermal pool at N=1/2/4, GPU radiosity pool at N=1/2/4, gray radiosity at N=1/4), Bennu, Keplerian asteroid, Europa | Yes: repository configs plus data shipped in `data/` |
| `validate-baseline-flagship-2026-08-15` | The two site-local flagship arms: 509,337-face lunar mesh, N=2 and N=4 GPU thermal | No: needs the moon3d mesh and corrected view-factor CSR, which live outside the repository on this machine |

The split is not cosmetic. A colleague with a fresh checkout can run the first
suite and cannot run the second; a suite that pretends otherwise sends them
hunting for a 4 GB file that was never distributed.

The flagship arms carry a pinned environment, and it is part of the baseline.
They were run with `PTM_ILLUM_TRUE_NORMALS=1`, `PTM_ILLUM_RISE_HABS=0`,
`PTM_HMATRIX_VF_REPLACE=<corrected CSR>` (the production illumination and
view-factor settings) and `PTM_MAX_TIMESTEPS_PER_DISPATCH=50` (a wedge
mitigation that also fixes the GPU dispatch chunking). Reproducing their bytes
requires the same environment; the exact strings are in the flagship manifest.
The 25 GB `.vf` is loaded, never rebuilt, and then replaced by
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
| `moon_crater80N_vfbuild_sparse` | `examples/validation/baseline_crater80N_vfbuild_sparse.yaml` | The sparse view-factor builder, matrix built not loaded; the only arm that can see a change to the near-field estimator |
| `moon_crater80N_vfbuild_hmatrix` | `examples/validation/baseline_crater80N_vfbuild_hmatrix.yaml` | The H-matrix builder, matrix built not loaded, including its ACA low-rank compression |
| `moon_crater80N_radoff_cpu` | `examples/tutorial04_crater80N_radoff.yaml` | Illumination plus CPU thermal with radiation disabled |
| `moon_crater80N_gputhermal_n{1,2,4}` | `examples/validation/baseline_crater80N_gputhermal.yaml` | GPU persistent-kernel thermal solver, the multi-GPU *thermal* pool, and the two-phase cadence that makes the 2026-08-15 defaults active |
| `moon_crater80N_radgpu_n{1,2,4}` | `examples/validation/baseline_crater80N_radgpu.yaml` | GPU radiosity solver and the multi-GPU *radiosity* pool (needs a sparse view-factor matrix; the GPU solver refuses an H-matrix) |
| `asteroid_bennu_spice_cpu` | `examples/quickstart_asteroid_bennu.yaml` | Generic-body thermophysics on a real SPICE ephemeris and an imported facet shape |
| `asteroid_keplerian_cpu` | `examples/quickstart_asteroid_diurnal.yaml` | Generic-body thermophysics on the analytic Keplerian propagator |
| `europa_diurnal_cpu` | `examples/quickstart_europa_diurnal.yaml` | Europa built-in body, primary-body infrared and reflected light, a full eclipse inside the output window |
| `moon509k_gputhermal_n{2,4}` | site-local (see the flagship manifest) | Production scale: 509,337 faces, corrected CSR view factors, 600 equilibration steps |

The generic-body arms are not optional. A passing lunar gate says nothing
about a change gated to the generic-body path; for such a change "Moon
byte-identical" is a tautology, because the gate excluded the Moon by
construction. Commit `5bfbdbe` scrambled the asteroid diurnal field while
passing the bowl-crater test, lunar byte-identity, and the seam check.

### View-factor builder arms

Added 2026-08-23 on `fix/hmatrix-symmetric-quadrature`. Every other arm in the
suite is structurally inert to a change in how view factors are computed:

| arm | why it cannot see a view-factor change |
|---|---|
| `asteroid_bennu_spice_cpu`, `asteroid_keplerian_cpu`, `europa_diurnal_cpu` | `radiation.enabled: false` |
| `moon_crater80N_radoff_cpu` | radiation off |
| `moon_crater80N_coupled_cpu` | loads a cached H-matrix `.vf` |
| `moon_crater80N_radgpu_n{1,2,4}`, `moon_crater80N_gputhermal_n{1,2,4}` | load a prebuilt `.vf` |

This is not hypothetical. The 2026-08-23 symmetric near-field quadrature fix
moves sunlit crater-rim faces by up to +13.45 K, and came out 5/5
byte-identical against the CPU arms of this suite. A green suite there was a
tautology, not do-no-harm evidence, and it would have said exactly the same
thing about a fix that was wrong.

The two new arms differ from every other one in a single essential way:
`radiation.view_factor_file: ''`, so the matrix is built on every run. One
takes the sparse builder, one the H-matrix builder, because PTM3D has two
in-engine view-factor implementations with two different near-field order rules
and `ptm_thermal_sim` chooses between them by a memory threshold as well as by
config. An arm covering one says nothing about a run that took the other.

Confirm from the log that the matrix was built, not loaded:

```
grep -E 'Using sparse view factor matrix|Building H-matrix' <log>   # must appear
grep -E 'Loading view factor'                              <log>   # must NOT
```

Sensitivity is checked directly. Flipping
`radiation.nearfield_quadrature_order` from 3 to 1 (which makes the near field
a bare M1 centroid estimate) moves 9 of 32 output files on both arms
(`temperature_avg/min/max`, `indirect_solar_max`, and their `_raw` counterparts);
`ptm_validate_baseline.py equivalence` returns `NOT EQUIVALENT`, exit 1. That
negative control is what makes a green verdict from these arms mean something.

These two arms are not in `validate-baseline-2026-08-15`, and their bytes
are not comparable to it. They are new configs, and they run physics that
changed on purpose: the near-field estimator was one-sided until 2026-08-23 and
read −21.7 percent on edge/vertex-touching pairs. Their baselines are banked
separately, at the branch and commit named in their own manifest, with the
attribution in `notes`; currently `postmerge-baseline-2026-09-05` at `c137564`
(section 2.1). Before that they were banked in `postmerge-baseline-2026-08-24` at
`51bf168` (deleted 2026-09-03), which superseded `vfunify-stage2-2026-08-23` at `fc57d58`. Do not
merge them into the 2026-08-15 manifest and do not retag it.

Both arms are byte-identical between `fc57d58` and `51bf168`, 32/32 each, at 48
threads and `PTM_ARCH_NATIVE=OFF`. That was verified rather than assumed, and the
verification is one command: `git diff fc57d58 51bf168` touches exactly one file,
`.github/workflows/ci.yml`. "The branch was merged unchanged" is a claim about
the tree, and a merge commit is where such a claim quietly stops being true.

### Two-phase and sparse-radiosity arms

The two `examples/validation/` configs are not variants for variety's sake:

- `baseline_crater80N_gputhermal.yaml` sets
  `equilibration_samples_per_diurnal_cycle: 24`, which makes the equilibration
  cadence differ from the output cadence and the run two-phase. That is the
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

Everything under a run's `statistics/` directory is compared, recursively, including
the `_raw/` subdirectory written by `output.write_raw_stats`. The historical
six-file comparison (temperature and illumination, average/minimum/maximum) is
a subset: the current suite compares 16 to 32 files per arm depending on
which channels the arm produces: 16 for the radiation-off and generic-body
arms, 22 for the coupled CPU crater, 26 for the gray arms, 32 for the arms that
also write raw statistics. 330 files across the 13 portable arms.

Excluded, by name, because their bytes legitimately vary between runs:
`provenance.txt` (records wall-clock time and the invoking command),
`manifest.json`, `.DS_Store`.

Two pitfalls:

- `statistics/*.bin` begins with a `uint32` header, so a naive floating-point
  comparator can report "identical" on files that differ. Compare bytes.
- `effective_albedo.bin` legitimately carries `NaN`, so a naive comparator
  reports "differs" at a maximum absolute difference of zero. Again: compare
  bytes, not values.

---

## 5. Running the suite

`scripts/ptm_validate_baseline.py` has four subcommands: `run` executes the
arms, `bank` digests them into a manifest, `verify` compares against one, and
`equivalence` compares two run directories directly. The arm set itself lives
in `scripts/validation_arms.json`, which is tracked, so a clone has it.

A bare `verify --manifest M` used to compare the baseline with itself.
With no `--arm` it resolved the runs to compare as the manifest's own recorded
run directories, re-digested the banked runs, and reported `330/330 files
byte-identical`, `VERDICT: BYTE-IDENTICAL`, on every arm, from any tree, at any
commit, including GPU arms this document calls unreproducible. That was
the command printed here until 2026-09-05 and the documented
fast path. It now exits 2 and prints no verdict at all. So does
any `--arm` that points at a banked directory. If you find that command written
down anywhere, it is stale; the tool will tell you so rather than pass.

```
# THE ONE-COMMAND GATE: run every banked arm fresh and compare it (section 2.1).
# --keep leaves the fresh runs behind for inspection; omit it and they go into a
# temporary directory that is removed afterwards.
python3 scripts/ptm_validate_baseline.py verify \
    --manifest ~/projects/ptm-validation-bank-2026-09-05/baseline_manifest_2026-09-05.json \
    --rerun --repo . --binary build/bin/ptm_thermal_sim \
    --run-root <a scratch directory> --keep

# Or run the arms yourself, then compare what you produced.
python3 scripts/ptm_validate_baseline.py run \
    --repo . --binary build/bin/ptm_thermal_sim \
    --run-root <scratch>/runs \
    --inputs-dir ~/projects/ptm-validation-bank-2026-09-05/inputs
python3 scripts/ptm_validate_baseline.py verify \
    --manifest ~/projects/ptm-validation-bank-2026-09-05/baseline_manifest_2026-09-05.json \
    --repo . --binary build/bin/ptm_thermal_sim \
    --arm moon_crater80N_radoff_cpu=<scratch>/runs/moon_crater80N_radoff_cpu  ...

# Only the CPU arms (a machine with no GPU, or a CPU-only build)
python3 scripts/ptm_validate_baseline.py run --group cpu ...

# Compare two finished run directories directly (N=1 versus N=4, CPU versus GPU)
python3 scripts/ptm_validate_baseline.py equivalence <run_a> <run_b>

# The 2026-08-15 bank, for the flagship arms it still owns (section 2.2).
# That manifest predates schema 3, so it carries no recipes and CANNOT be
# --rerun; produce the runs yourself and pass --arm. The tool says so.
python3 scripts/ptm_validate_baseline.py verify \
    --manifest ~/projects/validate_rebaseline_2026-08-15/bank/baseline_manifest_flagship.json \
    --repo . --binary build/bin/ptm_thermal_sim --arm <name>=<rundir>
```

`run` and `verify --rerun` do more than invoke the engine. Each arm's log is
checked for the line its dispatch rung prints (`Multi-GPU dispatch across 2
device(s)`, `ACTIVE RUNG: single-GPU radiosity` against `multi-GPU pool`,
`Sparse view factors computed:` against `Loaded view factors from:`), and an
arm that did not take its intended path is a failure, not a note. Before
this was mechanical it was a habit, and a GPU arm that silently fell back to
the CPU still wrote a full statistics directory and still compared
byte-identical to another arm that fell back the same way.

`verify` keys a `--binary` by its build directory name, so
`build_gpu/bin/ptm_thermal_sim` and `build/bin/ptm_thermal_sim` are different
keys. Point it at the build the manifest was cut from, or it reports that it
checked neither the linked libraries nor the build flags.

`verify` reports provenance before bytes. If a config file's bytes have
changed since the baseline was cut, if the engine commit differs, or if a
linked dynamic library differs, it says so first, because a byte comparison
across any of those changes answers a different question than the one you
asked, and the byte verdict alone would mislead you.

Until the branch that adds this document is merged, a bare checkout of
`ecf635a` does not contain `examples/validation/`. Running `verify --repo`
against such a checkout correctly reports those three configs as `GONE` and
returns a MISMATCH *even though all 330 files are byte-identical*; the tool is
refusing to certify a comparison whose reference configs it cannot see. That is
working as designed, not a false alarm. Verify from a checkout that has the
configs.

The engine commit is read from each run's `provenance.txt`, not from the
repository's `HEAD`: the sha is compiled in at build time, so a manifest banked
after an unrelated commit would otherwise credit the arms to an engine that
never ran them. If two arms report different engine shas, `bank` refuses
outright, since a mixed manifest blesses nothing.

---

## 6. What the baselines were built with, and what invalidates them

### Embree

Byte-identity is a property of the whole toolchain, not of the engine
binary. `ptm_thermal_sim` links Embree dynamically, so an Embree swap moves
results with the engine binary's checksum completely unchanged.

These baselines were built against Homebrew Embree 4.3.3, `EMBREE_MAX_ISA=SSE2`:

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
`-DEMBREE_ROOT=$HOME/opt/embree-4.3.3-avx512` changes lunar bytes: measured on
2026-08-15, 75 of 260 files differ on a lunar site run, worst face-frame
difference 0.00907 K. Europa and Bennu are byte-identical across the swap.

A comparison run under a different Embree is therefore not a byte-identity comparison against these baselines.
Either build both arms against the same Embree, or state the ISA and treat
lunar differences at the 0.01 K scale as toolchain, not regression. Every run
records `embree_library` and `embree_max_isa` in its `provenance.txt`, and the
configure step prints the ISA; read them rather than assuming.

`test_bowl_crater` cannot referee this. It builds its view factors without
Embree (`test_bowl_crater.cpp`), so passing it is a do-no-harm check, never a
discriminating test of an Embree change. The Embree-exercising bowl target is
`bowl_crater_ptm`.

### The engine's own libraries, and what an executable md5 does not cover

The canonical build is `BUILD_SHARED_LIBS=ON`, so `ptm_thermal_sim` is a thin
driver and the physics lives in `libthermal.dylib`, `libillumination_model.dylib`,
`libptm_gpu.dylib` and their siblings. A thermal-only change therefore moves
`libthermal.dylib` and leaves the executable's md5 untouched. `otool -L`
reports all of them as `@rpath/...`, which `os.path.exists` cannot resolve, so
until 2026-09-05 the manifest recorded `md5: null` for every one of them and the
provenance block was blind to exactly the code most likely to change. `bank` now
resolves `@rpath` through the binary's own `LC_RPATH` entries and records each
library's md5. Quote the dylib md5 alongside the executable's, or a "no change"
claim is not one.

Do not compare a dylib md5 across two worktrees. Two builds of identical
source at different absolute paths produce dylibs that differ in exactly 48
bytes (the `LC_UUID` and an `LC_RPATH` carrying the build directory) at the
same file size. Measured 2026-09-04 and again 2026-09-05 on `libthermal.dylib`:
`022838ab…` against `82de252e…`, 270,312 bytes each, `cmp -l` reports 48
differing bytes. The manifest therefore records two digests per library:

| Field | Semantics |
|---|---|
| `md5` | the whole file. The strict comparison, and the right one when both builds live at the same path |
| `text_md5` | the `__TEXT,__text` section only, address column dropped. Invariant to `LC_UUID` and `LC_RPATH`, so it is the right one across worktrees |

On the 48-byte case above the two `text_md5` values are equal
(`96237b973279b7de3b4eb68df0d6c7f8`), which is what makes "same code, built
elsewhere" a statement the tool can make rather than a claim the operator has to
be trusted on. `verify` prints which of the two it is.

### `PTM_ARCH_NATIVE`, the unrecorded build flag

The 2026-08-15 arms were built with `PTM_ARCH_NATIVE=ON`
(`ptm-rebaseline/build_cpu/CMakeCache.txt`), which turns on AVX2 and AVX-512 code
generation. Neither the manifest nor this document recorded it. Every later
comparison was made from a worktree configured `OFF` (the documented onboarding
combination), and every one of them read a byte difference that nobody could
attribute.

Measured 2026-08-24. Rebuilding `ecf635a` with `PTM_ARCH_NATIVE=ON` reproduces the
banked radiation-off arm 16/16 byte-identical; rebuilding the same commit with
`OFF` does not. The physical size of the `ON`-versus-`OFF` difference at `ecf635a`
is one face, maximum 3.2e-20: a single denormal in six flux files, no
temperature file touched. That is the whole difference, enough to break byte-identity,
far too small to mean anything, and exactly the kind of difference a byte gate is
supposed to surface rather than absorb.

On `51bf168` the flag is inert on all five portable CPU arms (`OFF` == `ON`,
22/22 and 16/16 × 4) and moves only the float64 `statistics/_raw/*_sum.bin`
accumulators on the two `vfbuild` arms: 5 of 32 files, maximum relative
difference 1.6e-12, summation reassociation under AVX-512. The `_raw` sums are
the same files the thread-count change moves, for the same reason.

Record the ISA configuration next to the thread count, and compare only
across a matched pair. Both `postmerge-baseline-2026-08-24` manifests name
their flag; an older manifest that does not name one was almost certainly cut at
`ON`, and a byte verdict against it from an `OFF` build is unattributable.

### The zero-occlusion view-factor window

Any `.vf` built by the engine sparse builder between `adc2aa5` (2026-07-20)
and `0b41b3b` fired its occlusion rays in the mesh frame into a scene
`RadiationExchangeManager::initEmbree` had recentered, so no candidate pair was
ever rejected as blocked. On the 12,800-face crater that is 0 of 1,307,376
pairs and up to −6.52 K on `temperature_max`.

`validate-baseline-2026-08-15`'s five radiosity arms (`radgpu_n{1,2,4}` and
`radgpu_gray_n{1,4}`) load exactly such a matrix, which is
why they cannot simply be re-verified: their input was wrong, so their bytes bless
the wrong physics. The seven copies of it are quarantined, not deleted, at
`~/projects/quarantine_zero-occlusion-vf_2026-08-24/` with a reversal manifest,
and a corrected replacement was banked at
`<workdir>/postmerge_2026-08-24/crater80N_sparse.vf` until that directory was deleted
on 2026-09-03 (section 2.1). The current bank carries its own sparse matrix under `inputs/`.

A missing `.vf` is not a hazard here: `ptm_thermal_sim` announces
`Early View Factor Computation`, builds the matrix, prints its occlusion counts,
and saves it to the named path (`src/cpp/main/ptm_thermal_sim.cpp`, the block
guarded on `!vf_file_exists`). The hazard is a `.vf` that exists and is silently
wrong, which is what version 1 of the sparse format now announces on load.

### The 2026-08-15 defaults in the banked bytes

These baselines were cut *after* the radiation-gate defaults flipped. Three
behaviors are therefore in the banked bytes:

| Default | Kill switch |
|---|---|
| Equilibration-step radiosity skip, on | `PTM_RAD_SKIP_EQ_STEPS=0` |
| Output-window face selector, on | `PTM_RAD_SELECT_OUTPUT_WINDOW=0` |
| Picard coupling cap raised from 10 to 16 | `radiation.max_coupling_iterations: 10` |

Both environment variables are tri-state: unset means stand down and print
why, `=1` means require the path and fail loudly if it is unavailable, `=0`
means off. Only `1/true/on/yes` and `0/false/off/no` parse; anything else is
fatal at configuration-validation time.

Comparing across the default flip requires the kill switches on both arms.
Without them, the flip's own intended change (measured at the time: 6 of 22
files, 33 of 509,337 faces, at most 2.5e-5 K) is indistinguishable from a
regression.

### Other things that invalidate a comparison

- Rebuilding between arms. Build once and run both arms with that binary, or
  rebuild both arms identically.
- A changed reference config. `verify` checks each config's md5 against the
  banked value and fails loudly. A config edit does not make the baseline wrong;
  it makes the baseline *about something else*.
- A different machine. The GPU arms are pinned to this host's four AMD
  W6800X Duo compute devices. Nothing here claims cross-machine byte-identity.
- A different thread count. Several OpenMP reductions in the solver and in
  the statistics accumulators sum in thread-partition order, so the same binary
  on the same config at 8 and at 48 threads writes different bytes. Measured
  2026-08-23 on `moon_crater80N_vfbuild_sparse`: at 48 threads 32/32 files match
  the `vfunify-stage1-2026-08-23` manifest; at 8 threads, 31/32 (one file,
  `statistics/_raw/temperature_sum.bin`; nothing else changed). Since that
  date the engine records `omp_max_threads` in `provenance.txt`, `bank` copies
  it into the manifest, and `verify` reports a thread-count change before
  the byte verdict. An arm banked by an older engine records no thread count and
  `verify` says so; treat its byte verdict as unattributable until it is re-cut.

---

## 7. Equivalences validated at `ecf635a`

Measured on 2026-08-15 with the binaries and Embree recorded above. Every GPU
arm's log was checked to confirm the intended dispatch rung was actually active,
so none of these is a fallback tautology.

| Equivalence | Result |
|---|---|
| Same binary, same config, run twice (CPU crater) | 22/22 files byte-identical; the engine is self-reproducible, which is the precondition for everything else here |
| GPU thermal pool, N=1 versus N=2 | 32/32 byte-identical |
| GPU thermal pool, N=1 versus N=4 | 32/32 byte-identical |
| GPU radiosity pool, N=1 versus N=2 | 32/32 byte-identical |
| GPU radiosity pool, N=1 versus N=4 | 32/32 byte-identical |
| Gray radiosity, N=1 versus N=4 | 26/26 byte-identical (see the caveat below) |
| Flagship 509k GPU thermal, N=1 versus N=2 versus N=4 | 32/32 byte-identical, twice independently (see section 9) |

`gpu_devices: 4` is therefore evidence-backed on this machine for both pools.
No default was changed; the shipped default remains `1`. Whether production
should move to 2 or 4 is a separate decision.

### A known mode difference in gray radiosity

In gray (non-band-separated) radiosity the single-GPU path and the pool
configure the solver differently: the single-GPU path clears `warm_start` only
when two-band is active, while the pool clears it unconditionally
(`radiation_exchange.cpp`, "Cross-batch warm-start has no meaning across
devices"). A warm-started and a cold-started iteration can stop at different
iterates, so N=1 and N>1 may legitimately differ in gray mode.

Measured on the 12,800-face crater in gray mode, N=1 versus N=4: 26/26
byte-identical. The hazard is real in the code and did not materialize here.
The crater's view-factor spectral radius is 0.134, so its iteration converges
almost immediately from either start; a high-spectral-radius mesh (the 509k
flagship is 0.776) is where this would show up, and no gray-mode arm exists at
that scale. Treat a gray-mode N=1-versus-N>1 difference as a known mode
difference to report, not as a suite failure, and not as evidence that
two-band is affected, because two-band clears `warm_start` on both paths.

---

## 8. What this suite does not cover

The value of a gate is bounded by what it touches. The suite does not cover the following.

- Scientific accuracy. Byte-identity is a *refactor* contract. It says the
  numbers did not move; it says nothing about whether they were right. Diviner
  cross-validation (`scripts/divval_orchestrate.py`) is the data-anchored
  check, and the Ingersoll bowl-crater test is the analytic one.
  Concretely: the flagship arms use a deliberately short shape (600
  equilibration steps, 20 output steps) built for cost and equivalence probing,
  and `scripts/ptm_qa.py thermal` fails its never-lit-face coherence check
  on them, both on this baseline's run (standard deviation of the minimum
  temperature, 21.65 K against a 3.0 K threshold) *and* on the earlier
  independent run of the same recipe at `d279eca` (13.95 K). The failure is a
  property of the recipe, not of the baseline. Never quote a flagship arm as a
  converged thermal field.
- The two site-local flagship arms. They were not re-run at `51bf168`, so
  `validate-baseline-flagship-2026-08-15` stands unre-baselined and its own
  view factors are unaudited by this work. Production scale (509,337 faces, and
  a view-factor spectral radius of 0.776 against the crater's 0.134) is
  therefore ungated on current `main`.
- `test_bowl_crater` is a smaller gate on a CPU build than on a GPU build, and
  it does not say so. It reports `PASS: 12 of 12 verdicts (0 skipped)` when
  built with `PTM_ENABLE_GPU=OFF` and `PASS: 16 of 16 verdicts (0 skipped)` with
  `ON`; the four missing verdicts are the GPU-against-CPU radiosity comparison,
  which is compiled out rather than skipped. `0 skipped` reads as full coverage
  in both cases. Quote the build configuration alongside the verdict.
- The exosphere, camera, and merge-chunks binaries. Only `ptm_thermal_sim`
  is exercised here.
- Illumination on the GPU. Every arm traces shadow rays on the CPU with
  Embree. `--use-gpu-illumination` is untested by this suite.
- Chunked runs and warm start. The `initial_state_file` /
  `final_state_file` path and `ptm_merge_chunks` have their own equivalence
  requirement and no arm here.
- Cross-machine reproducibility, as above.
- Gray-mode radiosity at production spectral radius, as above.
- Any Europa or asteroid GPU path. The generic-body and Europa arms are all
  CPU; a GPU change gated to those bodies would pass this suite untested.

## 9. Flagship scale: the N=1/2/4 ladder

Measured on the 509,337-face lunar mesh with the corrected view-factor CSR
loaded (nnz 495,163,704; never rebuilt), 600 equilibration steps and
20 output steps, `gpu_batch_size: 4000`, `PTM_MAX_TIMESTEPS_PER_DISPATCH=50`.

| Arm | Wall clock | GPU kernel device-time | Kernel dispatches |
|---|---|---|---|
| N=1 | 1834 s | 1588.6 s | 687 |
| N=2 | 1112 s | 1586.2 s | 687 |
| N=4 | 650 s | 1588.9 s | 687 |

Bytes: N=1 == N=2 == N=4, 32/32 identical. The N=2-versus-N=4 result was
obtained twice independently (once on a contended pair and once on the
clean pair), and byte-identity is not a timing property, so contention cannot
manufacture it.

Measured end-to-end speedups: 1.649× at N=2, 2.822× at N=4 (both against
N=1), and 1.711× for N=4 over N=2.

Two things make the speedup credible rather than merely observed:

- The device-time is the same on all three arms (1588.6 / 1586.2 / 1588.9 s,
  a spread of 0.17%) and so is the dispatch count (687). Every arm did the same
  arithmetic; the gain is parallel execution, not less work.
- The wall difference is attributable. Every other phase is device-count
  independent, and the one other phase the log times directly (the coupling's
  indirect-flux sweeps) is 50 s at N=2 and 51 s at N=4.

### The three-point scaling model

There is no per-phase timer in the log, so isolating the thermal phase requires
a model: `T(N) = F + C/N`, with `F` device-count independent. Two points
determine that model exactly, and the answer depends on which two:

| Points used | `C` (parallelizable) | `F` (fixed) |
|---|---|---|
| N=2 and N=4 | 1848 s | 188 s |
| N=1 and N=2 | 1444 s | 390 s |

They disagree by 28%, so `1/N` is a good approximation and not exact. A
least-squares fit over all three points gives `C` = 1559 s and `F` = 289 s (a
parallel fraction of 0.850), with residuals of −0.8%, +3.9% and −4.4%.

Therefore the 1.711× for N=4 over N=2 is measured; a thermal-phase-only
speedup near 2× is model output, not a measurement. Quote the first, and if
you quote the second, say it is derived and name the residuals. Had this been
stopped at two arms, the report would have carried `C` = 1848 s as a fact.

Contention is the dominant error source in any of these numbers. The same
N=2 arm measured 1297 s while an unrelated job held the GPU and 1112 s alone: a
14% penalty, consistent with the 18.5% previously documented on this host for
identical code paths. Never quote a speedup taken beside another job.
