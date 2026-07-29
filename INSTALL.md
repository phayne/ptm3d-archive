# Installing PTM3D

PTM3D is a C++/OpenCL planetary thermophysical model. The **portable core is CPU-first**
(threaded); the GPU (OpenCL) path is an **optional accelerator** where it exists. This guide
covers a source build on Linux and macOS. Windows and conda-forge packaging are planned
(portability plan Phases 2 and 4).

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
(low-res bodies, asteroids). The 500k micro-crater class is workstation/cluster-scale. See
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
and point CMake at it (`-Dembree_DIR=` or the `EMBREE_INCLUDE_DIRS`/`EMBREE_LIBRARIES` cache vars).

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
prefixes. It FATAL_ERRORs with an actionable message if it can't find `libcspice.a`.

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

## 4. Verify the build (correctness gates)

```bash
# Radiation gate (Ingersoll bowl crater — analytic):
cmake --build build --target test_bowl_crater -j && ./build/bin/test_bowl_crater
# Spectral equilibration:
cmake --build build --target test_equilibration_dt -j && ./build/bin/test_equilibration_dt
# Phase R runtime-robustness unit tests (no deps):
c++ -std=c++17 src/cpp/tests/test_machine_profile.cpp -o /tmp/tmp_mp && /tmp/tmp_mp
python3 scripts/test_ptm_run_guard.py
```

## 5. Run something

See `TUTORIAL.md` for a 10-minute laptop walkthrough (bowl crater → 1k-face asteroid diurnal run
→ small crater with radiosity). A single run:
```bash
./build/bin/ptm_thermal_sim --config examples/<a-relative-example>.yaml
```

## 6. Known caveats (from the portability audit)

- Many shipped `examples/*.yaml` still carry absolute machine paths (being relativized to
  `PTM_DATA_ROOT` — `docs/PORTABILITY-AUDIT.md` §1a). Prefer the tutorial's configs until then.
- `BUILD_TESTS=ON` currently references a nonexistent top-level `tests/` dir; build the individual
  test targets directly (as in §4).
- The GPU path is macOS+AMD-tested; the Phase R machine profile + solver self-chunking guard against
  the driver command-buffer watchdog. On an un-profiled host the solver uses conservative defaults
  and warns.
