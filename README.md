```
    ╔═══════════════════════════════════════════════════════════════╗
    ║    *      .    *     \   ||    //      * .   .      *         ║
    ║       .      *     \  \  ||   //    //       .             .  ║
    ║  *      .       \   \  \  ☀  //    //      .      *           ║
    ║     .         *  \   ...-'''''-...//    *         .           ║
    ║           _ ... .-'''''   PTM3D     ``''-.....,__      .      ║
    ║   __,.-''    Planetary Thermophysical Model 3D     `'-..., _  ║
    ║  ═══════════════════════════════════════════════════════════  ║
    ║      Coupled Thermal-Illumination for Planetary Bodies        ║
    ╚═══════════════════════════════════════════════════════════════╝
```

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/std/the-standard)

**PTM3D** predicts surface and subsurface temperatures on planetary bodies by
coupling ray-traced illumination (Intel Embree + NASA SPICE ephemerides) with a
1D Crank-Nicolson thermal solver and full inter-face thermal radiation exchange
on arbitrary 3D terrain. It is built for physical accuracy and computational speed across a range of planetary thermal environments, from permanently shadowed craters to icy satellite surfaces and asteroids.

![Lunar polar crater temperatures](docs/figures/temp_3d_native_twoband_v3.png)
*A lunar south-polar crater in PTM3D: (a) diurnal-mean surface temperature on
the native terrain mesh; (b) the warming contributed by inter-face radiation
exchange (two-band radiosity) — crater walls and floors heated by scattered
sunlight and thermal emission from surrounding terrain.*

## What it does

- **Bodies**: Moon, Mercury, Europa — plus any
  body via Keplerian elements, SPICE kernels, or JPL Horizons
- **Illumination**: Embree ray-traced shadows with finite solar disk penumbra,
  eclipses, and primary-body reflected + IR flux (e.g. Earthshine, Jupiter-shine)
- **Thermal**: Crank-Nicolson conduction with T-dependent k(T), cp(T), adaptive
  depth grids, and FFT spectral equilibration for fast spin-up
- **Radiation exchange**: face-to-face view factors, stored sparsely or in
  hierarchically compressed ("H-matrix") form to stay affordable at large face
  counts, driving a two-band (solar/IR) radiosity solver coupled to the
  thermal model
- **Ice**: subsurface ice-stability depth (vapor diffusion), and a full Europa
  exosphere/ice-evolution research module
- **Meshes**: built-in icosphere generator, PLY/OBJ/VTK import, DEM-to-mesh and
  synthetic crater tools; ParaView-compatible outputs
- **Performance**: threaded CPU core everywhere; optional OpenCL GPU
  acceleration for the thermal, radiosity, and ray-tracing stages

## Validation

- **Ingersoll et al. (1992)** analytic bowl-crater radiation: `test_bowl_crater`
  reproduces the analytic wall view factor to **0.49%** and satisfies view-factor
  reciprocity to machine precision. It is the gate every radiation change has to
  pass; the CPU build and this gate are wired into the repository's CI workflow
  ([`.github/workflows/ci.yml`](.github/workflows/ci.yml), Linux + macOS)
- **Lunar Diviner** radiometer: RMS <4 K against measured bolometric temperatures
- **Hayne et al. (2017)** 1D thermophysical model: reproduces the lunar
  equatorial diurnal cycle (noon ~388 K, night ~99 K) —
  ![see the comparison](docs/figures/lunar_validation.png)

## Get started

Everything below is **CPU-only** and runs on a stock 16 GB laptop; the GPU is an
optional accelerator, never required. Tested on macOS (AppleClang) and Linux
(GCC); on Windows use WSL2 and follow the Linux path.

### 0. Get the code

```bash
git clone git@github.com:phayne/ptm3d.git            # SSH — needs a key on your GitHub account
# or: git clone https://github.com/phayne/ptm3d.git  # HTTPS — asks for a GitHub token as password
cd ptm3d
```

A `Repository not found` error means the GitHub account you are authenticated as
cannot see the repository — ask Paul to add your GitHub username, then retry.

### 1. Prerequisites

| Need | Version | Where it comes from |
|---|---|---|
| CMake | ≥ 3.10 (developed against 3.31) | `apt install cmake` · `brew install cmake` |
| C++ compiler | C++17 — GCC 10+, Clang 12+, AppleClang | `build-essential` · Xcode Command Line Tools |
| Embree | **4.x** (not 3.x) | `apt install libembree-dev` · `brew install embree` |
| Eigen3, CFITSIO, yaml-cpp, OpenMP | any recent | `apt install libeigen3-dev libcfitsio-dev libyaml-cpp-dev libomp-dev pkg-config` · `brew install eigen cfitsio yaml-cpp libomp pkg-config` |
| CSPICE | N0067+ | **not in apt/brew** — download from NASA NAIF, [INSTALL.md §2](INSTALL.md) |
| Python | 3.8+, standard library only | `scripts/fetch_kernels.py`, `fetch_data.py`, `ptm_qa.py`, `ptm_mem_estimate.py` |
| OpenCL | optional | only for the GPU build |

Per-platform commands, including the CSPICE download: [INSTALL.md](INSTALL.md).

### 2. Build the portable CPU core

```bash
export CSPICE_ROOT=/path/to/cspice     # the directory containing lib/libcspice.a
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPTM_ENABLE_GPU=OFF \
      -DPTM_ARCH_NATIVE=OFF -DCSPICE_ROOT="$CSPICE_ROOT"
cmake --build build --target ptm_thermal_sim -j
```

`-DPTM_ARCH_NATIVE=OFF` builds a portable baseline; drop it for a
`-march=native` build tuned to the machine you compile on. Configure takes
~12 s and ends with a summary block naming every library it resolved — read it
before building; a missing CSPICE stops there with a message saying what to
set. The build itself is ~50 s at `-j8` on a workstation.

**Which build did you get?** GPU support is opt-in: a `cmake` line without
`-DPTM_ENABLE_GPU=ON` produces a CPU-only binary *even on a machine with a
supported GPU*. Confirm it, don't assume it:

```bash
# (a) at configure time CMake states which core it is building:
#     -- GPU acceleration OFF - building the portable CPU-first core
#     -- GPU acceleration enabled (backend: OPENCL)
# (b) from the finished binary — no output means CPU-only:
otool -L build/bin/ptm_thermal_sim | grep -i opencl     # macOS
ldd    build/bin/ptm_thermal_sim | grep -i opencl       # Linux
```

For the GPU build pass `-DPTM_ENABLE_GPU=ON`, then profile the host once with
`./build/bin/ptm_thermal_sim --calibrate-gpu`. A CPU-only binary handed a config
with `thermal.use_gpu: true` refuses at startup (`FATAL [gpu]`, exit 1) instead
of ray-tracing the whole run and failing at the thermal stage.

### 3. Fetch SPICE kernels (once)

```bash
python3 scripts/fetch_kernels.py --bodies moon   # Moon + Mercury core, ~44 MB
```

The bare `fetch_kernels.py` also pulls the 1.1 GB Galilean ephemeris that the
shipped `europa.tm` needs — ~1.15 GB total; use `--bodies moon` for this
walkthrough. Downloads resume after an interruption and every file is SHA-256
verified. Run `ptm_thermal_sim` **from the repository root**: the shipped
metakernels use repo-root-relative paths.

### 4. Run the first simulation

```bash
./build/bin/ptm_thermal_sim --config examples/quickstart_moon_diurnal.yaml
```

One lunation of a global Moon on a 1280-face icosphere — 5 s wall on a 48-core
workstation; it threads over cores, so allow longer on a laptop. A healthy log
names the mesh it actually loaded, the thread count it chose (yours will differ),
and finishes with:

```
Mesh created with 642 vertices and 1280 faces
Using 48 OpenMP threads (auto)
Simulation completed in 5.030 seconds (5030 ms)
[Statistics] Exported to output/quickstart_moon_diurnal/statistics
```

Results land in `output/quickstart_moon_diurnal/` (~10 MB): per-timestep
`temperature/`, `illumination/` and flux directories, the `mesh.ply` they index,
and `statistics/` with the per-face extremes over the run. Expect
`temperature_max.bin` to peak at **385.0 K** (equatorial noon) and
`temperature_min.bin` to bottom at **66.3 K** (a polar face), with the mean
per-face minimum ~90.8 K. Every `statistics/*.bin` starts with a 4-byte
face-count header, then `float32` per face:

```bash
python3 -c "
import struct, array
b = open('output/quickstart_moon_diurnal/statistics/temperature_max.bin','rb').read()
n = struct.unpack('<i', b[:4])[0]; a = array.array('f'); a.frombytes(b[4:4+4*n])
print(n, 'faces; Tmax range', round(min(a),1), '-', round(max(a),1), 'K')"
# -> 1280 faces; Tmax range 182.0 - 385.0 K
```

For a 3D view of the same output, point the [GUI](#gui) at the run directory, or
open `mesh.ply` in ParaView. Field-by-field walkthrough:
[docs/tutorials/06_interpreting_outputs.md](docs/tutorials/06_interpreting_outputs.md).

### 5. Confirm the install is sound

```bash
python3 scripts/ptm_qa.py thermal output/quickstart_moon_diurnal  # -> verdict: PASS
cmake --build build --target test_bowl_crater -j     && ./build/bin/test_bowl_crater
cmake --build build --target test_equilibration_dt -j && ./build/bin/test_equilibration_dt
```

`ptm_qa.py thermal` checks the run's own statistics for NaNs, uncomputed faces
and range violations. `test_bowl_crater` is the analytic radiation gate (look
for `Error: 0.49...%` and `Result: PASSED`); `test_equilibration_dt` gates the
spectral spin-up and prints `[test_equilibration_dt] PASS`. All three are
CPU-only and take seconds. There is no `ctest`/`pytest` — the tests are
standalone executables, listed in [INSTALL.md §5](INSTALL.md).

### Science data (not needed for the quickstart)

Topography, property and albedo maps, and the validation crater meshes are too
large for git; they are archived on Zenodo and fetched on demand:

```bash
python3 scripts/fetch_data.py --list          # the catalog: 6 core datasets, ~1.5 GB
python3 scripts/fetch_data.py --core          # fetch all of them (resumable, SHA-256 verified)
export PTM_DATA_ROOT=/big/disk/ptm3d-data     # optional: keep the data outside the clone
```

Relative `data/...` paths in a config resolve against `$PTM_DATA_ROOT` first and
the repository's own `data/` second (an absolute path in the config always
wins). Catalog, provenance and licenses: [data/README.md](data/README.md).

Next: [TUTORIAL.md](TUTORIAL.md) is a 10-minute walkthrough that adds a polar
crater with radiosity, and [docs/tutorials/](docs/tutorials/) goes deeper.

## Documentation

| Where | What |
|-------|------|
| [INSTALL.md](INSTALL.md) | Dependencies, build options, SPICE kernel data, verification |
| [TUTORIAL.md](TUTORIAL.md) | 10-minute laptop walkthrough |
| [docs/tutorials/](docs/tutorials/) | Numbered tutorials: diurnal runs, crater radiosity, interpreting outputs |
| [docs/USER_GUIDE.md](docs/USER_GUIDE.md) | CLI reference, YAML schema, output formats, architecture, Python tools |
| [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) | Every `PTM_*` environment variable, with status tags |
| [docs/configuration.md](docs/configuration.md) | Complete YAML configuration reference |
| [docs/thermal-model.md](docs/thermal-model.md) · [docs/illumination-model.md](docs/illumination-model.md) · [docs/radiation-exchange.md](docs/radiation-exchange.md) | Physics and numerics of each stage |
| [docs/KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md) · [docs/PORTABILITY-AUDIT.md](docs/PORTABILITY-AUDIT.md) | What to watch out for — **read this before your first production run** |
| [data/README.md](data/README.md) | Science-data catalog, provenance, licenses, `PTM_DATA_ROOT` |
| [gui/README.md](gui/README.md) | The desktop GUI |

### Which configuration directory to start from

**Start in [`examples/`](examples/), not `configs/`.** The 15 configurations at
the top level of `examples/` are the ones written to be read and run by someone
who did not build the repository; 12 of the 15 are CPU-only, and
`quickstart_moon_diurnal.yaml` is the smallest.

`configs/` (55 YAML files at its top level) and `examples/campaigns/` (74) are
the *internal* production inputs behind published runs. They are kept for
provenance, not as onboarding material: 47 of the 55 set `thermal.use_gpu: true`,
several point at meshes, DEMs or view-factor files that are not distributed, and
many carry `thermophysical:` keys that PTM3D does not read (the run warns and
names them — see [docs/KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md), PTM-67).
Copy one only if you already know what it references.

## Hardware guidance

The dominant memory cost is the stored illumination time series:
`faces × timesteps × 3 flux components × 8 bytes` (plus the view-factor matrix
when radiation is on). Estimate any run with
`python3 scripts/ptm_mem_estimate.py --faces N --steps M`.

| Mesh size | Steps | Resident memory | Runs on |
|----------:|------:|----------------:|---------|
| 1k–10k faces | 360 | ≤ 90 MB | any laptop |
| 100k faces | 360 | ~0.9 GB | 16 GB laptop |
| 100k faces | 1440 | ~3.5 GB | 16 GB laptop (borderline — stream) |
| 500k faces | 1440 | ~17 GB | workstation / cluster |
| 1M faces (high-res DEM site) | 360 | ~8.6 GB (+ sparse VF, ~8 GB†) | workstation; GPU recommended |

† Size of the sparse view-factor matrix when radiation exchange is on: about
8 bytes per stored matrix entry, and the number of radiatively coupled partner
faces per face is **terrain-dependent** — measured ~1,040 partners per face
(20 GB at 2.4M faces) on one polar site vs. 4.8 GB at 140k faces on a site
where each face couples to more of its surroundings. The ~8 GB figure scales
the former; budget from your own mesh with a small test build.

The GPU accelerates the thermal, radiosity and ray-tracing stages; it does not
reduce these memory requirements. See "Which build did you get?" above.

## GUI

An optional PyQt6 desktop GUI configures runs and visualizes results in 3D. It
is independent of the C++ build — install it into a virtual environment so its
dependencies (PyQt6, PyVista, matplotlib) stay off your system Python:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r gui/requirements.txt
cd gui && python run_gui.py
```

`run_gui.py`, launched from `gui/`, is the one supported entry point — details
and troubleshooting in [gui/README.md](gui/README.md).

## Citation

If you use PTM3D in your research, please cite it (machine-readable metadata in
[CITATION.cff](CITATION.cff)):

```bibtex
@software{ptm3d,
  author = {Hayne, Paul O.},
  title = {PTM3D: Planetary Thermophysical Model 3D},
  year = {2026},
  version = {v1.0.3},
  institution = {University of Colorado Boulder},
  url = {https://github.com/phayne/ptm3d}
}
```

The supporting science datasets (fetched by `scripts/fetch_data.py`) are
archived on Zenodo: concept DOI
[10.5281/zenodo.21693403](https://doi.org/10.5281/zenodo.21693403)
(v1: [10.5281/zenodo.21693404](https://doi.org/10.5281/zenodo.21693404), CC-BY-4.0).

Heritage methods: Hayne et al. (2017, [10.1002/2017JE005387](https://doi.org/10.1002/2017JE005387));
Hayne & Aharonson (2015, [10.1002/2015JE004887](https://doi.org/10.1002/2015JE004887));
Vasavada et al. (2012, [10.1029/2011JE003987](https://doi.org/10.1029/2011JE003987));
Paige et al. (2010, [10.1126/science.1187726](https://doi.org/10.1126/science.1187726)).
H-matrix view factors: Potter et al. (2023, [10.1016/j.jcpx.2023.100130](https://doi.org/10.1016/j.jcpx.2023.100130)).

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

[Sam Potter](https://github.com/sampotter) (H-matrix algorithms,
[fluxpy](https://github.com/sampotter/fluxpy)); NASA NAIF (SPICE); Intel
(Embree); the LRO Diviner team; NASA Lunar-VISE, Europa Clipper, and PSIE.

## Support

Questions, bug reports and feature requests go to the repository's **Issues**
tab (<https://github.com/phayne/ptm3d/issues>) so the answer is findable by the
next person. For a build or run problem, include your OS and compiler, the
CMake configure summary block, and the first ~40 lines of the run log — those
three almost always identify it. For anything you would rather not put in an
issue, contact Paul Hayne (paul.hayne@colorado.edu).
