# Installing PTM3D

PTM3D is a planetary thermophysical model written in C++. The **core model runs entirely on
ordinary CPUs** (using all available cores); the GPU (OpenCL) path is an **optional
accelerator** where suitable hardware exists. This guide
covers a source build on Linux and macOS. On **Windows, use WSL2** (Ubuntu) and follow the
Linux instructions — a native MSVC build is planned (portability plan Phase 2), as is
conda-forge packaging (Phase 4).

> **TL;DR (Linux/macOS, CPU-only):**
> ```bash
> cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPTM_ENABLE_GPU=OFF \
>       -DPTM_ARCH_NATIVE=OFF -DCSPICE_ROOT=/path/to/cspice
> cmake --build build --target ptm_thermal_sim -j
> ./build/bin/ptm_thermal_sim --help
> ```

## 1. Requirements

| Component | Minimum | Notes |
|-----------|---------|-------|
| CMake | 3.10+ | `/usr/local/bin/cmake` on the dev workstation |
| C++ compiler | C++17 (GCC 10+, Clang 12+, AppleClang) | MSVC support is Phase 2 |
| Eigen3 | 3.3+ | header-only |
| Embree | 4.x | ray tracing for illumination/view factors — stock packages are SSE2-only; see §3 "Embree ISA" |
| CSPICE | N0067+ | NASA NAIF SPICE toolkit (C) — **not** in apt/brew; see below |
| CFITSIO | any | camera/exosphere I/O (can be made optional) |
| yaml-cpp | any | config parsing |
| OpenMP | any | threaded CPU core (`libomp` on macOS) |
| OpenCL | optional | only for `PTM_ENABLE_GPU=ON` |

Minimum useful machine: a **16 GB laptop** runs real science on **1k–100k-face meshes**
(low-res bodies, asteroids). The 500k–1M-face class (micro-craters, high-resolution DEM
sites) is workstation/cluster-scale, GPU recommended. See
`docs/PORTABILITY-AUDIT.md` §4 for the memory scaling table, and estimate a run with:
```bash
python3 scripts/ptm_mem_estimate.py --faces 100000 --steps 360   # -> ~0.80 GiB, OK on 16 GB
```

## 2. Dependencies per platform

### Linux (Debian/Ubuntu)
```bash
sudo apt-get update
sudo apt-get install -y cmake g++ libomp-dev libeigen3-dev libcfitsio-dev \
                        libyaml-cpp-dev libembree-dev pkg-config
```
If `libembree-dev` is Embree 3 on your image, install Embree 4 from the official release tarball
and point CMake at it (the `-Dembree_DIR=` or `EMBREE_INCLUDE_DIRS`/`EMBREE_LIBRARIES` settings).

### macOS (Homebrew)
```bash
brew install cmake embree eigen cfitsio yaml-cpp libomp pkg-config
```
Homebrew's `embree` bottle is built SSE2-only for portability, which leaves measurable
illumination performance on the table on any AVX/AVX-512 host — see §3 "Embree ISA (performance)"
for how to check what you have and how to build a multi-ISA one alongside it.

### CSPICE (both platforms — manual)

CSPICE is distributed by NASA NAIF, not a package manager. It is shipped as a **prebuilt static
library, one package per CPU architecture** — so pick the package that matches the machine you
are building on:

| Build host | NAIF package directory | Download |
|---|---|---|
| Linux, x86_64 | `PC_Linux_GCC_64bit` | ~41 MB compressed |
| macOS, **Apple Silicon** (`uname -m` → `arm64`) | `MacM1_OSX_clang_64bit` | ~40 MB compressed |
| macOS, Intel (`uname -m` → `x86_64`) | `MacIntel_OSX_AppleC_64bit` | ~40 MB compressed |

The snippet below selects the right one for you:

```bash
case "$(uname -s)/$(uname -m)" in
  Linux/*)       NAIF_PKG=PC_Linux_GCC_64bit        ;;
  Darwin/arm64)  NAIF_PKG=MacM1_OSX_clang_64bit     ;;   # Apple Silicon (M1/M2/M3/M4)
  Darwin/x86_64) NAIF_PKG=MacIntel_OSX_AppleC_64bit ;;   # Intel Mac
  *) echo "No NAIF package mapping for $(uname -s)/$(uname -m)"; exit 1 ;;
esac
URL="https://naif.jpl.nasa.gov/pub/naif/toolkit//C/${NAIF_PKG}/packages/cspice.tar.Z"
curl -sSL "$URL" -o cspice.tar.Z && gzip -d cspice.tar.Z && tar xf cspice.tar
# CMake's find_library wants libcspice.a; NAIF ships lib/cspice.a:
cp cspice/lib/cspice.a cspice/lib/libcspice.a
export CSPICE_ROOT="$PWD/cspice"
```

**On Apple Silicon, verify the architecture before you build.** CMake's `find_library` only checks
that `libcspice.a` *exists*; it does not check what it was compiled for. Handing an arm64 build the
Intel (`MacIntel_OSX_AppleC_64bit`) package therefore **configures cleanly and then fails at the
link step**, because the linker silently ignores an archive built for another architecture and
every SPICE symbol (`_furnsh_c`, `_spkezr_c`, …) comes out undefined for `arm64`. One command
settles it:

```bash
lipo -info "$CSPICE_ROOT/lib/libcspice.a"
# -> ... is architecture: arm64     correct on an Apple Silicon Mac
# -> ... is architecture: x86_64    wrong package — re-download MacM1_OSX_clang_64bit
```

(An Intel Mac wants the opposite answer, `x86_64`. Running an arm64 build under Rosetta is not
supported here; build natively.)

PTM3D resolves CSPICE from `-DCSPICE_ROOT=`, then `$CSPICE_ROOT`/`$CSPICE_DIR`, then common
prefixes. If it cannot find `libcspice.a`, the configure step stops with a clear error message
saying what to set.

## 3. Configure & build

### Portable CPU core (recommended default)
```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPTM_ENABLE_GPU=OFF \
  -DPTM_ARCH_NATIVE=OFF \
  -DCSPICE_ROOT="$CSPICE_ROOT"
cmake --build build --target ptm_thermal_sim -j
```
`-DPTM_ARCH_NATIVE=OFF` builds a portable baseline (no `-march=native`); drop it (default ON) for a
locally-tuned build. `PTM_ENABLE_GPU=OFF` is the default on a fresh checkout.

### GPU-accelerated build (OpenCL-capable hosts)
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPTM_ENABLE_GPU=ON -DCSPICE_ROOT="$CSPICE_ROOT"
cmake --build build --target ptm_thermal_sim -j
# One-time per host: profile the safe GPU dispatch limits (Phase R):
./build/bin/ptm_thermal_sim --calibrate-gpu     # writes ~/.ptm3d/machine_profile.json
```
On macOS the legacy `-DENABLE_GPU=ON` still works (it is the compatibility alias for
`PTM_ENABLE_GPU`). Apple Silicon has no OpenCL future — use the CPU core there.

#### GPU checklist before a long run (macOS)

A long PTM3D GPU run on macOS with AMD hardware can wedge the GPU hard enough to
power-cycle the machine. Three items, once per host:

1. **Disable sleep and display sleep.** A display or system sleep transition issues a
   GPU modeset, which is the documented trigger of the wedge; the last logged system
   event before the 2026-08-20 power-off was `Display is turned off`.
   ```bash
   sudo pmset -a disablesleep 1 displaysleep 0     # before a campaign
   sudo pmset -a disablesleep 0 displaysleep 30    # after
   ```
   The engine checks this itself and warns at startup if it is not set — it reads
   `pmset -g custom`, which needs no privileges, and it never changes a setting.
2. **Calibrate the dispatch ceiling** (`--calibrate-gpu`, above). Uncalibrated hosts
   fall back to conservative limits and say so.
3. **Use the chunked runner for anything long**, so a wedge is caught by the stall
   abort and the failed attempt's log survives the rerun:
   ```bash
   python3 scripts/run_chunked_simulation.py --config <config> \
       --total-days <days> --output-dir <dir>
   ```

Expect one thing to look different on a big radiation run: when the view-factor
replica leaves less than 2 GiB of VRAM headroom per device, the engine puts the
radiosity replica and the GPU thermal pool on **disjoint devices** and says so
on the `[Radiosity] ACTIVE RUNG` line (roughly +8 min on a 30-minute window).
It gives up GPU radiosity entirely only below a 0.25 GiB hard floor. Both
behaviors are configurable under `gpu:` and are macOS+AMD-gated.

If a run wedges: the log stops mid-phase while the process stays alive, and after the
reboot `pmset -g log | grep -i "shutdown cause"` reports a negative SMC code
(`-108`/`-128`). `log show` needs sudo; `pmset -g log` does not and keeps about a week.
The recovery procedure and the reasoning behind every guardrail — including which ones
are macOS+AMD-only — are in [docs/gpu-wedge-guardrails.md](docs/gpu-wedge-guardrails.md).

### Embree ISA (performance)

**A stock package-manager Embree is usually compiled SSE2-only, and on a modern x86 CPU that
costs real wall clock in PTM3D.** Embree compiles a separate BVH traversal kernel per SIMD
instruction set, fixed at *its* build time by `EMBREE_MAX_ISA`, and dispatches at run time to the
widest one it was built with. A redistributable package has to run on every CPU it might land on,
so it is built for the lowest common denominator: Homebrew's `embree` 4.3.3 bottle is
`EMBREE_MAX_ISA=SSE2`.

Ray traversal is roughly 83% of PTM3D's illumination phase, so this is not a rounding error.
Measured on the project's Mac Pro (Xeon W-3265M, Cascade Lake, AVX-512) on 2026-08-15, against
the **identical** Embree 4.3.3 source rebuilt multi-ISA:

| | SSE2-only (Homebrew) | multi-ISA (`AVX512`) |
|---|---|---|
| `rtcOccluded1`, traversal micro-benchmark | 1.00x | 1.22–1.45x |
| illumination phase, `calib120` arm, end to end | 13.12 s | **12.20 s (−7.0%)** |

The end-to-end figure is 18 paired interleaved repeats across two sessions (18/18 favoring
multi-ISA; t = 6.0 and 13.8). It is quoted with its limits: the thermal phase, which casts no
rays and therefore cannot be affected, was measured alongside as a negative control and came out
null (t = −0.8 and −1.0), which is what makes the illumination difference believable. A longer
709-step arm on the same machine could **not** resolve the difference at all (n = 6, sd 45 s) —
the control was equally contaminated there, so that is machine contention, not evidence against
the effect. Expect "a few percent off the illumination phase", not a headline speedup; the
traversal micro-benchmark ratio is not an end-to-end ratio.

#### How to tell which ISA you are running

Three independent checks, cheapest first:

1. **At run time** — every run prints one line to stderr before the first ray is cast:
   ```
   [Embree] 4.3.3  traversal kernels: AVX-512 (16-wide)  (native packets 4=yes 8=yes 16=yes)
   ```
   An SSE2-only library on an AVX host prints `SSE (4-wide)` and an explicit `[Embree] WARNING:`
   naming the cost. For Embree's own definitive target report, set `PTM_EMBREE_VERBOSE=1`.
   (This line reads Embree's native packet widths, which Embree gates on `EMBREE_RAY_PACKETS`
   as well as on the ISA — hence `-DEMBREE_RAY_PACKETS=ON` in the recipe below. A library built
   with packets off reports that it cannot tell, rather than guessing "SSE2".)
2. **At configure time** — CMake names the library it took and the kernels it carries:
   ```
   -- Embree: /Users/you/opt/embree-4.3.3-avx512/lib/libembree4.4.dylib
   -- Embree: version 4.3.3, max ISA AVX512, kernels: SSE2 SSE42 AVX AVX2 AVX512
   ```
   A `EMBREE_MAX_ISA=SSE2` library on an x86-64 host raises a CMake warning.
3. **From the library itself** — count the wide-register instructions; an SSE2 build has none:
   ```bash
   otool -tV /usr/local/Cellar/embree/4.3.3/lib/libembree4.4.dylib | grep -c ymm   # -> 0
   otool -tV "$HOME/opt/embree-4.3.3-avx512/lib/libembree4.4.dylib" | grep -c zmm  # -> nonzero
   # Linux: objdump -d <lib> | grep -c ymm
   ```

**After the fact**, every run's `provenance.txt` records which Embree produced it:
```
embree_library=/Users/you/opt/embree-4.3.3-avx512/lib/libembree4.4.dylib
embree_max_isa=AVX512
```
This is not redundant with `engine_binary_md5`. Embree is a dynamic library, so upgrading it — or
rebuilding the same prefix at a different `EMBREE_MAX_ISA` — changes results while leaving the
executable's checksum untouched.

#### Building a multi-ISA Embree

This does **not** replace or modify the package-manager install; it sits alongside it in its own
prefix, and nothing uses it until you pass `-DEMBREE_ROOT`.

```bash
curl -sSL -o embree-4.3.3.tar.gz \
  https://github.com/embree/embree/archive/refs/tags/v4.3.3.tar.gz
tar xf embree-4.3.3.tar.gz

cmake -S embree-4.3.3 -B embree-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$HOME/opt/embree-4.3.3-avx512" \
  -DEMBREE_MAX_ISA=AVX512 \
  -DEMBREE_RAY_PACKETS=ON \
  -DEMBREE_TASKING_SYSTEM=TBB \
  -DTBB_ROOT=/usr/local/opt/tbb \
  -DEMBREE_ISPC_SUPPORT=OFF \
  -DEMBREE_TUTORIALS=OFF
cmake --build embree-build -j
cmake --install embree-build
```

Notes: `EMBREE_MAX_ISA=AVX512` is a *maximum*, not a requirement — the build still emits SSE2
through AVX2 kernels and picks at run time, so the result stays correct on older CPUs.
`EMBREE_RAY_PACKETS=ON` is Embree's default and is kept explicit because the run-time ISA check
above reads packet widths. `EMBREE_ISPC_SUPPORT=OFF` and `EMBREE_TUTORIALS=OFF` only skip
components PTM3D does not use.
`TBB_ROOT` is macOS/Homebrew-specific; on Linux drop it if TBB is on the default search path.
Build time is ~25 min on 24 cores. Use a prefix that a scratch/worktree cleanup will not delete.

Then point PTM3D at it:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DEMBREE_ROOT="$HOME/opt/embree-4.3.3-avx512" \
  -DCSPICE_ROOT="$CSPICE_ROOT"
```

`EMBREE_ROOT` may also be an environment variable; `-Dembree_ROOT=` and `-DCMAKE_PREFIX_PATH=`
are equivalent CMake-native spellings. If `EMBREE_ROOT` is set but the search still resolves
elsewhere, the configure **stops** rather than silently building against the other library —
clear the CMake cache and check that `$EMBREE_ROOT/lib/cmake/embree-<version>/` exists.

> **Treat an Embree swap as a results change, not just a speed change.** Different ISA kernels
> make different floating-point association choices, so grazing-incidence occlusion verdicts can
> flip. Measured on the SSE2 → multi-ISA swap: Europa (Pwyll, ico6) and Bennu byte-identical, but
> a lunar polar DEM patch moved 75 of 260 output files with a worst face-frame |ΔT| of 0.0091 K.
> That is far inside any physical tolerance, but re-validate rather than assume.

## 4. SPICE kernel data

Building links against the CSPICE **library** (§2); *running* additionally needs SPICE
**kernel data** (ephemerides, orientation, leapseconds). These are large binary files that
are not kept in the repository — fetch them from NASA NAIF with the bundled Python script
(no extra Python packages required):

```bash
python3 scripts/fetch_kernels.py                  # default (auto): ~1.15 GB — see note
python3 scripts/fetch_kernels.py --bodies moon    # Moon + Mercury core only, ~44 MB
python3 scripts/fetch_kernels.py --bodies europa  # core + jup365.bsp (1.1 GB) for Europa
```

**The bare command downloads ~1.15 GB, not ~44 MB.** In its default `--bodies auto` mode the
script fetches the core set *and* whatever the metakernels present in `data/kernels/` require —
and `europa.tm` ships with the repository, so it also pulls the Galilean ephemeris
`jup365.bsp` (1.1 GB). That is deliberate (it is what the installed metakernels need, and it
stops a Europa run failing twice on the same missing kernel), and the script announces the
auto-add before downloading. If you only want the Moon/Mercury quickstart, pass
`--bodies moon`: `naif0012.tls` (5 KB) + `pck00011.tpc` (128 KB) + `de440s.bsp` (31.2 MB) +
`moon_pa_de440_200625.bpc` (12.3 MB) + `moon_de440_250416.tf` (19 KB) = ~44 MB.

This populates `data/kernels/` with the layout the binary expects:

```
data/kernels/
├── moon.tm  europa.tm  mercury.tm   # metakernel templates (shipped with the repo)
├── lsk/naif0012.tls                 # leapseconds (required even in Keplerian mode)
├── pck/pck00011.tpc                 # body orientation/shape constants
├── pck/moon_pa_de440_200625.bpc     # lunar principal-axes orientation
├── fk/moon_de440_250416.tf          # MOON_PA / MOON_ME frame definitions
└── spk/de440s.bsp                   # planetary ephemeris
```

Interrupted downloads resume where they left off, every file is verified against a known
checksum (SHA-256), and re-running the script skips files that are already complete.

Two things to know:

- The kernel directory is built into the binary as `SPICE_KERNELS_DIR` (a CMake setting,
  default `<source-dir>/data/kernels`). Override at configure time with
  `-DSPICE_KERNELS_DIR=/abs/path` if you keep kernels elsewhere.
- The shipped metakernels use paths **relative to the repository root**
  (`PATH_VALUES = ('data/kernels')`), which SPICE interprets relative to the directory the
  program is started from — so **run `ptm_thermal_sim` from the repository root** (as all
  examples and tutorials do), or edit `PATH_VALUES` to an absolute path.

Small bodies driven by JPL Horizons (`generic_body` configs) cache their own ephemerides
under `data/horizons_cache/` and need only `lsk/naif0012.tls` from the above.

**Science datasets** (lunar topography, property maps, albedo maps, validation meshes) are
fetched the same way with `python3 scripts/fetch_data.py --core` (~1.7 GB) — see
`data/README.md` for the catalog, and `$PTM_DATA_ROOT` in `docs/ENVIRONMENT.md` to keep
the data outside the repository. The quickstart example needs only the SPICE kernels.

## 5. Python environment (QA, notebooks, plots)

The engine and the correctness gates need no Python packages. The QA checker
(`scripts/ptm_qa.py`), the tutorial notebooks and every plotting route do:

```bash
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

The root `requirements.txt` is the environment for the documented path — numpy,
matplotlib, PyYAML, JupyterLab, and SpiceyPy (needed by notebook 04 only). Two
narrower files exist for the parts of the repository beyond it:
`scripts/requirements.txt` (scipy, astropy and the rest of the analysis and
figure tooling) and `gui/requirements.txt` (PyQt6, PyVista and the desktop GUI).

## 6. Verify the build (correctness gates)

```bash
# Radiation gate (Ingersoll bowl crater — analytic):
cmake --build build --target test_bowl_crater -j && ./build/bin/test_bowl_crater
# Equilibration substep size (look for the '[test_equilibration_dt] PASS' verdict
# line; it gates the worst-case error at the production cadence against the 1 K
# target — the substep, not the spin-up length):
cmake --build build --target test_equilibration_dt -j && ./build/bin/test_equilibration_dt
# Runtime-robustness unit tests (no external libraries needed):
c++ -std=c++17 src/cpp/tests/test_machine_profile.cpp -o /tmp/tmp_mp && /tmp/tmp_mp
python3 scripts/test_ptm_run_guard.py
```

## 7. Run something

See `TUTORIAL.md` for a 10-minute laptop walkthrough. The quickstart (from the repo root,
after fetching kernels per §4):
```bash
./build/bin/ptm_thermal_sim --config examples/quickstart_moon_diurnal.yaml
```

**Take your configurations from `examples/`, not `configs/`.** The 21 files at the top of
`examples/` are the colleague-facing set and 20 of them are CPU-only. `configs/` (55 files)
holds study inputs kept for provenance: 47 of the 55 set `thermal.use_gpu: true`, so on the
CPU-only build of §3 they stop immediately with `FATAL [gpu]` (set `thermal.use_gpu: false`
to run them on the CPU), and several reference meshes, DEMs or view-factor files that are
not distributed.

## 8. Known caveats (from the portability audit)

- Files directly under `examples/` use relative paths and run from a clone. Files under
  `configs/` may reference datasets and machine-specific paths that are not distributed.
- `BUILD_TESTS=ON` currently references a nonexistent top-level `tests/` directory; build the
  individual test programs directly (as in §6).
- `rebuild.sh` in the repository root is a **macOS developer convenience script, not the
  supported build path**. It deletes the contents of `build/`, passes none of
  `-DPTM_ENABLE_GPU` / `-DPTM_ARCH_NATIVE` / `-DCSPICE_ROOT`, and so produces a **CPU-only**
  binary tuned to the build host's own CPU (`-march=native`), covering every default target
  rather than `ptm_thermal_sim` alone. It states all of that when it starts. Use §3 unless you
  specifically want those defaults.
- The GPU path has been exercised mainly on macOS with AMD hardware. A one-time per-machine
  calibration (`--calibrate-gpu`) plus automatic splitting of long GPU work protect against GPU
  driver timeouts, which can otherwise hang the machine; on an uncalibrated machine the solver
  falls back to conservative limits and says so in the log. The wedge guardrails in
  [docs/gpu-wedge-guardrails.md](docs/gpu-wedge-guardrails.md) are gated the same way: the ones
  that cost wall time or change a solver rung enforce on macOS+AMD and are advisory elsewhere,
  so they have **not** been exercised in their enforcing mode on any other platform.
