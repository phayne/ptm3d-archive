# Installing PTM3D

PTM3D is a planetary thermophysical model written in C++. The **core model runs entirely on
ordinary CPUs** (using all available cores); the GPU (OpenCL) path is an **optional
accelerator** where suitable hardware exists. This guide
covers a source build on Linux and macOS. On **Windows, use WSL2** (Ubuntu) and follow the
Linux instructions.

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
| CMake | 3.10+ | |
| C++ compiler | C++17 (GCC 10+, Clang 12+, AppleClang) | native Windows (MSVC) is not supported; use WSL2 |
| Eigen3 | 3.3+ | header-only |
| Embree | 4.x | ray tracing for illumination/view factors — stock packages are SSE2-only; see §3 "Embree ISA" |
| CSPICE | N0067+ | NASA NAIF SPICE toolkit (C) — **not** in apt/brew; see below |
| CFITSIO | any | camera/exosphere I/O (can be made optional) |
| yaml-cpp | any | config parsing |
| OpenMP | any | threaded CPU core (`libomp` on macOS) |
| OpenCL | optional | only for `PTM_ENABLE_GPU=ON` |

Minimum useful machine: a **16 GB laptop** runs real science on **1k–100k-face meshes**
(low-res bodies, asteroids). The 500k–1M-face class (micro-craters, high-resolution DEM
sites) is workstation or cluster scale. Estimate a run's memory with:
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

### DEM / DTM tools (optional — GDAL)

The engine does **not** need GDAL. It is needed only to build the two tools that turn a
GeoTIFF digital terrain model into a PTM3D mesh — `dtm_to_ply_delaunay` and
`dem_mesh_generator` — and to give the GUI dashboard's DEM preview its `osgeo` bindings.
If you never start from a DEM, skip this.

```bash
# macOS / Linux, recommended (installs the C++ library AND the Python bindings):
conda install -c conda-forge gdal

# macOS, Homebrew alternative (C++ library only; add the bindings separately):
brew install gdal
python3 -m pip install "GDAL==$(gdal-config --version)"

# Debian/Ubuntu:
sudo apt-get install -y libgdal-dev python3-gdal
```

CMake finds GDAL through its own config package or through `gdal-config`; a conda environment is
picked up automatically when `CONDA_PREFIX` is set, and `$HOME/anaconda3/envs/europa_viz` is
tried as a fallback hint. **Configure succeeds without GDAL** — it prints

```
CMake Warning: GDAL not properly configured. SKIPPING these targets: dem_mesh_generator,
dtm_to_ply_delaunay. ...
```

and builds everything else. If you install GDAL later, re-run `cmake -S . -B build` before
building; the skip is decided at configure time, not build time.

Then build the DEM tool (see §3):

```bash
cmake --build build --target dtm_to_ply_delaunay -j
```

`dtm_to_ply_delaunay` also uses the header-only [CDT](https://github.com/artem-ogre/CDT)
triangulation library, vendored at `third_party/CDT/` (MPL-2.0) — nothing to install.

The end-to-end DEM workflow, including what each command-line flag does and which parts of
the toolchain are **not** implemented, is documented in
[docs/dem-to-mesh-workflow.md](docs/dem-to-mesh-workflow.md).

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
# One-time per host: profile the safe GPU dispatch limits:
./build/bin/ptm_thermal_sim --calibrate-gpu     # writes ~/.ptm3d/machine_profile.json
```
On macOS the legacy `-DENABLE_GPU=ON` still works (it is the compatibility alias for
`PTM_ENABLE_GPU`). On Apple Silicon use the CPU build.

#### GPU checklist before a long run (macOS)

A long PTM3D GPU run on macOS with AMD hardware can wedge the GPU hard enough to
power-cycle the machine. Three items, once per host:

1. **Disable sleep and display sleep.** A display or system sleep transition issues a
   GPU modeset, a known trigger of the hang.
   ```bash
   sudo pmset -a disablesleep 1 displaysleep 0     # before a campaign
   sudo pmset -a disablesleep 0 displaysleep 30    # after
   ```
   The engine checks this itself and warns at startup if it is not set (it reads
   `pmset -g custom` and never changes a setting).
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
are macOS+AMD-only — are in [docs/dev/gpu-wedge-guardrails.md](docs/dev/gpu-wedge-guardrails.md).

### Embree ISA (performance)

Package-manager Embree builds are usually SSE2-only. On an AVX/AVX-512 CPU a multi-ISA build of
the same Embree version is a few percent faster in the illumination phase; it can also flip a
small number of grazing occlusion verdicts (hundredths of a kelvin). Every run prints which
Embree kernels it uses and records them in `provenance.txt`. How to check and how to build a
multi-ISA Embree: [docs/dev/embree-isa.md](docs/dev/embree-isa.md).

## 4. SPICE kernel data

Building links against the CSPICE **library** (§2); *running* additionally needs SPICE
**kernel data** (ephemerides, orientation, leapseconds). These are large binary files that
are not kept in the repository — fetch them from NASA NAIF with the bundled Python script
(no extra Python packages required):

```bash
python3 scripts/fetch_kernels.py                  # default (auto): ~1.23 GB — see note
python3 scripts/fetch_kernels.py --bodies moon    # Moon + Mercury core only, ~44 MB
python3 scripts/fetch_kernels.py --bodies europa  # core + jup365.bsp (1.1 GB) for Europa
python3 scripts/fetch_kernels.py --bodies bennu   # core + Bennu's ephemeris (80.7 MB)
```

**The bare command downloads ~1.23 GB, not ~44 MB.** In its default `--bodies auto` mode the
script fetches the core set *and* whatever the metakernels present in `data/kernels/` require.
Two of the shipped metakernels name a body kernel: `europa.tm` pulls the Galilean ephemeris
`jup365.bsp` (1.1 GB), and `bennu.tm` pulls `sb-101955-118_long.bsp` (80.7 MB). That is
deliberate (it is what the installed metakernels need, and it stops a Europa or Bennu run
failing twice on the same missing kernel), and the script announces each auto-add before
downloading. If you only want the Moon/Mercury quickstart, pass
`--bodies moon`: `naif0012.tls` (5 KB) + `pck00011.tpc` (128 KB) + `de440s.bsp` (31.2 MB) +
`moon_pa_de440_200625.bpc` (12.3 MB) + `moon_de440_250416.tf` (19 KB) = ~44 MB.

`sb-101955-118_long.bsp` is the only kernel here that does **not** come from NAIF. It is JPL
solution 118 for (101955) Bennu — the final OSIRIS-REx reconstruction, Farnocchia et al.
(2021) — over 1900-01-01 to 2135-10-01, downloaded from
`https://ssd.jpl.nasa.gov/ftp/xfr/`, which is the address JPL Horizons prints when it declines
to generate a Bennu SPK on request. Because that is a transfer directory rather than a
versioned archive path, the pinned SHA-256 is the only guarantee that the file has not changed
under its name; a hash mismatch there is worth reading before assuming a corrupt download.

This populates `data/kernels/` with the layout the binary expects:

```
data/kernels/
├── moon.tm  europa.tm  mercury.tm  bennu.tm   # metakernel templates (shipped with the repo)
├── lsk/naif0012.tls                 # leapseconds (required even in Keplerian mode)
├── pck/pck00011.tpc                 # body orientation/shape constants
├── pck/moon_pa_de440_200625.bpc     # lunar principal-axes orientation
├── fk/moon_de440_250416.tf          # MOON_PA / MOON_ME frame definitions
├── spk/de440s.bsp                   # planetary ephemeris
└── spk/sb-101955-118_long.bsp       # (101955) Bennu, 1900-2135 (--bodies bennu/auto)
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
fetched the same way with `python3 scripts/fetch_data.py --core` (about 1.4 GB) — see
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

## 6. Verify the build

```bash
# Quick radiation check (analytic bowl-crater view factors and reciprocity), CPU, seconds:
cmake --build build --target test_bowl_crater -j && ./build/bin/test_bowl_crater
#   -> [test_bowl_crater] PASS: 12 of 12 verdicts passed
# Spin-up substep size (not spin-up length):
cmake --build build --target test_equilibration_dt -j && ./build/bin/test_equilibration_dt
#   -> [test_equilibration_dt] PASS
# Unknown configuration keys and flags are reported:
python3 scripts/test_unknown_config_keys.py
```

A GPU build adds four GPU radiosity verdicts to `test_bowl_crater` (16 of 16). The tests are
standalone programs; there is no `ctest`.

## 7. Run something

Continue with the [tutorials](docs/tutorials/README.md); the first one runs the quickstart:
```bash
./build/bin/ptm_thermal_sim --config examples/quickstart_moon_diurnal.yaml
```

**Take your configurations from `examples/`, not `configs/`.** `examples/` holds configurations
written to be run from a clone (one of them, `europa_high_res.yaml`, needs a GPU build).
`configs/` holds study inputs kept for provenance: most set `thermal.use_gpu: true` (a CPU build
refuses them with `FATAL [gpu]`; set it to `false`), and several reference meshes, DEMs or
view-factor files that are not distributed.

## 8. Notes

- `BUILD_TESTS=ON` references a top-level `tests/` directory that does not exist; build the
  test programs individually as in §6.
- `rebuild.sh` is a macOS developer script that always reconfigures to a fixed configuration
  (GPU on, native ISA, shared libraries); use §3 for anything else. `./rebuild.sh --dry-run`
  prints what it would run.
- **Apple Silicon:** the portable CPU build (`-DPTM_ARCH_NATIVE=OFF`) is what continuous
  integration builds on `macos-latest` (arm64); the arm64 build fixes in this release were
  tested on an Apple M5 machine. Use the `MacM1_OSX_clang_64bit` CSPICE package (§2).
- The GPU path has been exercised mainly on macOS with AMD hardware; its guardrails are
  described in [docs/dev/gpu-wedge-guardrails.md](docs/dev/gpu-wedge-guardrails.md).
