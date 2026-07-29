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
| Embree | 4.x | ray tracing for illumination/view factors |
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

### CSPICE (both platforms — manual)
CSPICE is distributed by NASA NAIF, not a package manager:
```bash
# Linux:  PC_Linux_GCC_64bit ; macOS: MacIntel_OSX_AppleC_64bit
URL=https://naif.jpl.nasa.gov/pub/naif/toolkit//C/PC_Linux_GCC_64bit/packages/cspice.tar.Z
curl -sSL "$URL" -o cspice.tar.Z && gzip -d cspice.tar.Z && tar xf cspice.tar
# CMake's find_library wants libcspice.a:
cp cspice/lib/cspice.a cspice/lib/libcspice.a
export CSPICE_ROOT="$PWD/cspice"
```
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

## 4. SPICE kernel data

Building links against the CSPICE **library** (§2); *running* additionally needs SPICE
**kernel data** (ephemerides, orientation, leapseconds). These are large binary files that
are not kept in the repository — fetch them from NASA NAIF with the bundled Python script
(no extra Python packages required):

```bash
python3 scripts/fetch_kernels.py                  # Moon + Mercury set (~45 MB)
python3 scripts/fetch_kernels.py --bodies europa  # also jup365.bsp for Europa (large)
```

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

## 5. Verify the build (correctness gates)

```bash
# Radiation gate (Ingersoll bowl crater — analytic):
cmake --build build --target test_bowl_crater -j && ./build/bin/test_bowl_crater
# Spectral equilibration:
cmake --build build --target test_equilibration_dt -j && ./build/bin/test_equilibration_dt
# Runtime-robustness unit tests (no external libraries needed):
c++ -std=c++17 src/cpp/tests/test_machine_profile.cpp -o /tmp/tmp_mp && /tmp/tmp_mp
python3 scripts/test_ptm_run_guard.py
```

## 6. Run something

See `TUTORIAL.md` for a 10-minute laptop walkthrough. The quickstart (from the repo root,
after fetching kernels per §4):
```bash
./build/bin/ptm_thermal_sim --config examples/quickstart_moon_diurnal.yaml
```

## 7. Known caveats (from the portability audit)

- Configs under `examples/campaigns/` are internal production-campaign artifacts with
  machine-specific absolute paths and are **not runnable from a clone** (see the README
  there). Everything directly in `examples/` uses relative paths.
- `BUILD_TESTS=ON` currently references a nonexistent top-level `tests/` directory; build the
  individual test programs directly (as in §5).
- The GPU path has been exercised mainly on macOS with AMD hardware. A one-time per-machine
  calibration (`--calibrate-gpu`) plus automatic splitting of long GPU work protect against GPU
  driver timeouts, which can otherwise hang the machine; on an uncalibrated machine the solver
  falls back to conservative limits and says so in the log.
