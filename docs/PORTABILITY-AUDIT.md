# PTM3D Portability Audit (Phase 0)

**Date:** 2026-07-21 · **Tree:** repository root at `3e3baae`,
branch `feat/portability-phaseR`.
**Scope:** read-only inventory of the release tree. Findings are `file:line` with a severity and
a disposition. This is the audit input to the portability work (Linux CPU build, CI, path
portability), not the fix itself.

Severity: **P0** blocks a non-Mac / non-this-host build or run · **P1** wrong/silent on another host ·
**P2** hygiene / reproducibility.

---

## 1. Hardcoded absolute paths

### 1a. Shipped example configs (P1 — the plan's known follow-up)
`267` of the `examples/*.yaml|*.ptm3d` files embed a machine-specific absolute path
(`/Users/…`, an `OneDrive-…` path, or `/Volumes/…`):

```
grep -rlE '/Users/|OneDrive|/Volumes/' examples/ | wc -l   ->  267
```

These will not resolve on any other machine (or even this one after the OneDrive→`~/projects`
migration). Phase-1 disposition: relativize to `PTM_DATA_ROOT` / config-relative resolution
(see §4) and ship a small curated set of RELATIVE example configs; quarantine the rest as
non-shipped campaign artifacts.

### 1b. Compiled-in default paths in SOURCE (P0/P1 — ship in the binary)
- **`src/cpp/illumination/illumination_model.h:602`** —
  `std::string kernels_path_ = "/Users/paha3326/software/cspice/data";`
  A **default member initializer**: every build hardcodes this user's CSPICE data dir as the
  fallback kernel path. On any other host the SPICE furnsh silently uses a nonexistent dir.
  **P0** for portability. Fix: default empty + resolve from `$CSPICE_DATA` / `PTM_DATA_ROOT` /
  config, and fail loud if unresolved.
- **`src/cpp/camera/camera_sim_main.cpp:362`** — `config.kernels_path = "/Users/paha3326/software/cspice/data";`
  Same class, camera binary. **P1**.
- **`src/cpp/thermal/thermal_model.cpp:1811`** — debug dump `"/tmp/thermal_debug.txt"` (debug-gated). **P2**.
- **`src/cpp/exosphere/test_equilibrium_bias.cpp:139`, `test_pentagon_bias.cpp:326`,
  `test_smooth_kernel_real_data.cpp:270-271`** — OneDrive `mesh.ply` paths in test fixtures. **P2**
  (tests, not shipped runtime) but they break the exosphere test build-and-run on CI; parameterize
  via env/argv before wiring these into the CI battery.

### 1c. `/tmp` scratch conventions (P1 on non-writable-/tmp hosts)
- **`src/cpp/radiation/hmatrix_view_factor.cpp:1174`** —
  `ooc_path = "/tmp/hmatrix_" + std::to_string(getpid());` (H-matrix out-of-core scratch).
  The plan explicitly flags this. Fix: honor `$TMPDIR` (POSIX) then `/tmp`; Windows has no `/tmp`.
- **`src/cpp/radiation/benchmark_hmatrix.cpp:392`** — `"/tmp/test_hmatrix.bin"` (benchmark). **P2**.

---

## 2. macOS-isms

- **`src/cpp/gpu/opencl_config.h:13-17`** — OpenCL header split is **already portable**:
  `#ifdef __APPLE__ #include <OpenCL/opencl.h> #else #include <CL/cl.h> #endif`. Positive finding;
  Linux OpenCL builds get the right header. (GPU is optional/OFF for the portable core anyway.)
- **`#ifdef __APPLE__` platform branches** in `fileio/memory_mapped_io.cpp:24,309`,
  `radiation/view_factor.cpp:16,31,55`, `radiation/hmatrix_view_factor.cpp:21,43,76,4526,4549`,
  `exosphere/exosphere_physics.h:24` — these guard `aligned_alloc`/`posix_memalign`/OpenMP-header
  differences. Each needs its `#else` Linux branch **verified to compile** in Phase 1 (they appear
  to have non-APPLE branches; confirm on GCC + Clang). **P1 (verify)**.
- No `xcrun`, `-framework`, Metal, or CoreFoundation usage in source. Good.
- Build-script mac-isms (`brctl`/Finder/OneDrive hydration) live in `scripts/` runners and are
  benign on Linux (they no-op or are unused in the CPU path); audit the campaign scripts before
  advertising them cross-platform. **P2**.

## 3. OpenCL-isms outside `gpu/`

All OpenCL usage outside `gpu/` is in **`src/cpp/main/ptm_thermal_sim.cpp`** (`ThermalOpenCLConfig`,
`ThermalOpenCLSolver`, `opencl_config.work_group_size = 64` at lines 4051/4143/4092/5091/5154…) and
is **guarded by `#ifdef PTM_GPU_OPENCL`** — it compiles out cleanly in the CPU-only (portable-core)
build. No stray OpenCL types leak into the illumination/radiation/thermal libraries. **P1 (verify
the CPU-only TU compiles with the GPU blocks elided)** — this is a Phase-1 build gate, not a rewrite.

## 4. Memory assumptions + scaling table

The full absorbed-illumination series is held resident: `n_faces × n_steps × components` float64
(direct + primary-reflected + primary-IR ⇒ ~3 components) before the thermal solve. The knob
`PTM_RAM_ILLUM_MAX_GB` (read in the illumination path; a large shape model may need `=40`) triggers a streaming
path above the cap. This was sized on the 384 GB workstation. Scaling law and 16 GB-laptop guidance:

| faces | steps | components | resident (float64) | laptop verdict |
|------:|------:|-----------:|-------------------:|----------------|
| 1e3   | 360   | 3          | ~8.6 MB            | trivial |
| 1e4   | 360   | 3          | ~86 MB             | fine |
| 1e5   | 360   | 3          | ~864 MB            | fine on 16 GB |
| 1e5   | 1440  | 3          | ~3.5 GB            | borderline; stream or reduce cadence |
| 5e5   | 360   | 3          | ~4.3 GB            | workstation-class |
| 5e5   | 1440  | 3          | ~17 GB             | exceeds 16 GB — stream / chunk |
| 1e6   | 360   | 3          | ~8.6 GB            | large DEM site — workstation, GPU recommended; sparse VF adds ~8 GB (terrain-dependent¹) |

¹ Sparse-VF footprint = nnz × ~8 B with a terrain-dependent partner count:
measured ~1,040 partners/face (2.5B nnz, ~20 GB) at 2.4M faces on one polar
site, vs. 4.8 GB at 140k faces on a denser-coupled site. The 1e6 figure scales
the former linearly.

Phase-1 disposition: a **memory estimator that computes this product and WARNS (or refuses w/o
`--allow-oversubscribe`) before the big allocation**, defaulting the streaming cap sensibly on
≤16 GB hosts (query physical RAM). The VF (nnz × ~12 B) has its own budget; laptop guidance is
sparse-VF only, ≤100k faces, ≤360 steps resident (per the plan).

## 5. Hardware-tuned constants (never assume this host)

- **`opencl_config.work_group_size = 64`** hardcoded at every solver construction in
  `ptm_thermal_sim.cpp` (4051, 4143, 5091, 5154, …). W6800X-tuned. **P2** (GPU-only path).
- **`gpu_batch_size` default `0 = auto = 4 × gpu_devices`** (`config_parser.h:75,285`,
  `radiation_config.h:81`). The Phase-R **machine profile** now supplies a per-host
  `recommended_batch_size`; wire defaults to consume it. **P1**.
- **Dispatch/watchdog ceilings** — previously implicit; now first-class in
  `gpu/machine_profile.h` (`max_timesteps_per_dispatch`, `safe_dispatch_ceiling_s`) + enforced by
  the R2 self-chunking. Conservative default 300 timesteps/25 s if un-profiled.
- **Thread counts** — defaults are `0 = auto` (`config_parser.h:47` thermal `threads`, `:261`
  `hmatrix_threads`; `ptm_thermal_sim.cpp:8312` `rad_hmatrix_threads`), and `viewshed_analyzer.cpp:193`
  uses `std::thread::hardware_concurrency()`. **No hardcoded 48-thread default was found** (the `48`
  hits are quadrature `Np=48` and `48 B/tri`, unrelated). Phase-1 task: confirm every `0 = auto`
  path resolves to `hardware_concurrency()`, not a buried constant. **P1 (verify)**.
- `eq-cadence` defaults and `--gpu-inter-batch-pause-ms` are hardware-tuned knobs; document defaults.

## 6. Compiler assumptions

- **`-march=native`** for Intel Mac (`CMakeLists.txt:226`) and Linux x86-64 (`:231`). Fine for a
  local source build; **breaks binary redistribution and can SIGILL on a CI runner whose CPU lacks a
  used ISA extension**. Phase-1: gate behind an option (`PTM_ARCH_NATIVE`, default ON for local, OFF
  for CI/redistributable), and let CI pick a portable baseline (e.g. `-mavx2` or none). **P1**.
- **`-mcpu=apple-m1`** hardcoded for arm64 (`:222`) — prefer `-mcpu=native`/`apple-a14+` so M2/M3/M4
  aren't pinned to M1 scheduling. **P2**.
- **AVX2/AVX-512 `option(... ON)`** (`CMakeLists.txt:128-129`) default ON — AVX-512 is absent on many
  laptops/CI; the code guards with `__AVX512F__` etc., but the default-ON option must degrade
  gracefully (detect, don't assume). **P1 (verify on a non-AVX512 host)**.
- Compiler autodetect (`CMakeLists.txt:1-60`) already honors `CC/CXX/PTM_PREFERRED_CXX` and skips
  override under MSVC — good cross-compiler groundwork. **OK**.
- `AppleClang` is the only compiler exercised here; GCC + Clang-on-Linux are unproven until the
  CI matrix runs (Phase 1). Expect `-Werror`-class and `<cstdint>`/`<memory>` include gaps.

## 7. Dependency inventory

| Dep | How found today | Portable? | Phase-1 disposition |
|-----|-----------------|-----------|---------------------|
| **Embree 4** | `find_package(embree 4 QUIET)` → pkg-config → **hardcoded `/usr/local/lib/libembree4.dylib`** fallback (`CMakeLists.txt:288-306`) | yes | keep find_package; drop the mac fallback or make it a `find_library`; optional FetchContent pin |
| **CSPICE** | **hardcoded `CSPICE_ROOT=/Users/paha3326/software/cspice`** (`CMakeLists.txt:328`) + `find_library` | yes (source) | **P0**: no default user path; require `-DCSPICE_ROOT=` or `$CSPICE_ROOT`, fail with a clear message (it already FATAL_ERRORs if unfound — good) |
| **OpenMP** | `find_package(OpenMP REQUIRED)` + **mac libomp paths `/opt/homebrew`, `/usr/local`** (`:248-256`) | yes | keep the mac special-case guarded by APPLE; on Linux `find_package(OpenMP)` just works; make OpenMP an option (`PTM_ENABLE_OPENMP`) |
| **Eigen3** | `find_package(Eigen3 REQUIRED)` (`:316`) | yes | OK; header-only, optional FetchContent pin |
| **CFITSIO** | `find_package(CFITSIO REQUIRED)` (`:284`) | yes | make optional (camera/exosphere only) so the core builds without it |
| **yaml-cpp** | pkg-config / find_package / homebrew-path fallback (`:388-435`) | yes | keep; drop homebrew hardcoded fallback |
| **OpenCL** | via `ENABLE_GPU` → `gpu/CMakeLists.txt` autodetect (HIP/CUDA/OpenCL) | optional | already `option(ENABLE_GPU OFF)`; portable core needs none |
| **Python/pybind11** | `option(BUILD_PYTHON_BINDINGS OFF)` (`:500`) | optional | OK, off by default |

Self-contained FFT+PCG for the Fourier seed (plan constraint) — **keep dependency-free (no FFTW)**;
not yet present in this tree (lands with the Fourier work).

## 8. GPU persistent-solver architecture note (found during Phase R/R4)

The GPU thermal path is **batch-outer / time-inner**: for each face-batch it runs the persistent
kernel over ALL timesteps (`solvePersistentMultipleCycles`), then downloads that batch's final
state, then moves to the next batch (`ptm_thermal_sim.cpp:4478-4562`, assembled into
`assembled_final_T`, written once at `:4584`). Consequence: **there is no wall-clock moment at which
all faces share the same output step**, so true "checkpoint every N output steps" (R4) of a full-mesh
state is architecturally impossible here — a partial dump carries un-solved faces at 0 K and is a
resume hazard (the zero-column runaway guarded at `:4609-4614`). Loss is bounded today by the
runner's process-level chunk split (`run_yearlong_wedge_safe.py --stage-b-nchunks`). Per-output-step
checkpointing belongs to a time-outer CPU-first solver, or to a solver restructure.
The config key `thermal.checkpoint_every_output_steps` is plumbed and **warns loudly** if set on
the GPU path rather than silently no-op'ing (`ptm_thermal_sim.cpp` PHASE-3 mapping).

## 9. Phase-1 priority ordering (from this audit)

1. **P0**: remove hardcoded `CSPICE_ROOT` default (`CMakeLists.txt:328`) and the compiled-in
   `kernels_path_` (`illumination_model.h:602`).
2. **P1**: CMake options (`PTM_ENABLE_GPU` alias, `PTM_ENABLE_OPENMP`, `PTM_ARCH_NATIVE`); make
   CFITSIO optional; drop homebrew/`/usr/local` hardcoded fallbacks.
3. **P1**: `PTM_DATA_ROOT` + config-relative path resolution; relativize the curated example set.
4. **P1**: memory estimator + warning (§4); consume `recommended_batch_size` from the profile.
5. **P1**: verify Linux GCC+Clang compile of the `__APPLE__` `#else` branches and the CPU-only TU.
6. **CI** ratchet (ubuntu+macos, CPU): `test_bowl_crater` + 3-case mesh + exosphere unit + a 1k-face
   asteroid smoke.
