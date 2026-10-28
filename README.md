# PTM3D: Planetary Thermophysical Model 3D

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

PTM3D computes surface and subsurface temperatures of airless bodies on 3D terrain.
It couples ray-traced illumination (Intel Embree, NASA SPICE geometry), a 1D heat-conduction
column under every mesh face, and face-to-face exchange of scattered sunlight and thermal
infrared. It runs on ordinary CPUs; an OpenCL GPU is optional.

![Moon3D testbed crater: mean temperature and the warming from radiation exchange](docs/figures/moon3d_testbed_tavg_coupling.png)

*Moon3D testbed crater scan (509,337 faces, about 3 mm resolution) placed at the IM-2 landing
site (84.79°S): (a) time-averaged surface temperature over one synodic month with
face-to-face radiation exchange; (b) the warming that radiation exchange adds relative to an
otherwise identical run without it. From the Moon3D revision (JGR Planets, Fig. 12).*

## Install

CPU build on macOS or Linux (Windows: use WSL2 and the Linux steps). Full details, including
CSPICE, which is not in any package manager: [INSTALL.md](INSTALL.md).

```bash
brew install cmake embree eigen cfitsio yaml-cpp libomp pkg-config        # or the apt line in INSTALL.md
export CSPICE_ROOT=/path/to/cspice                                         # NAIF CSPICE, see INSTALL.md
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPTM_ENABLE_GPU=OFF -DPTM_ARCH_NATIVE=OFF \
      -DCSPICE_ROOT="$CSPICE_ROOT"
cmake --build build --target ptm_thermal_sim -j
```

## First run

From the repository root:

```bash
python3 scripts/fetch_kernels.py --bodies moon                  # SPICE kernels, about 44 MB
./build/bin/ptm_thermal_sim --config examples/quickstart_moon_diurnal.yaml
```

One lunation of a 1,280-face global Moon, after an 80-cycle spin-up: about 30 s on four
cores. The per-face results land in `output/quickstart_moon_diurnal/statistics/`, and the
hottest face reaches 385.0 K (equatorial noon). The first tutorial reads these files and
explains every line of the run log: [docs/tutorials/01_quickstart.md](docs/tutorials/01_quickstart.md).

## Documentation

Read in this order:

| Page | What it covers |
|---|---|
| [INSTALL.md](INSTALL.md) | Dependencies, build options, SPICE kernels, science data, verification |
| [docs/tutorials/](docs/tutorials/README.md) | Quickstart, crater radiation exchange, output files, GUI, and four Jupyter notebooks (Moon, Europa, asteroids) |
| [docs/USER_GUIDE.md](docs/USER_GUIDE.md) | Command line, output formats, Python package |
| [docs/configuration.md](docs/configuration.md) | Every configuration key and its default |
| [docs/KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md) | What the model does not do; read before a production run |
| [docs/VALIDATION.md](docs/VALIDATION.md) | Comparisons with analytic solutions, Diviner and Apollo data |
| [docs/thermal-model.md](docs/thermal-model.md), [docs/illumination-model.md](docs/illumination-model.md), [docs/radiation-exchange.md](docs/radiation-exchange.md) | Physics and numerics of each stage |
| [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) | Environment variables, including the opt-in fine-mesh illumination settings |
| [data/README.md](data/README.md) | Science datasets, provenance and licenses |
| [gui/README.md](gui/README.md) | The desktop GUI |
| [CHANGELOG.md](CHANGELOG.md) | Changes between releases |

## What it does

- **Bodies:** the Moon, Mercury and Europa with built-in properties; any other body through
  orbital elements, SPICE kernels or JPL Horizons.
- **Illumination:** ray-traced shadows with a finite solar disk, eclipses, and light and
  thermal emission from a primary body (Earthshine, Jupiter-shine).
- **Heat conduction:** Crank-Nicolson columns with temperature-dependent conductivity and heat
  capacity and depth-dependent density.
- **Radiation exchange:** face-to-face view factors (sparse or hierarchically compressed),
  two-band (visible and infrared) radiosity coupled to the thermal solution, and optional
  sub-facet roughness (beaming).
- **Products:** per-face temperature and flux statistics, per-step frames, subsurface depth
  maps, ice-stability depth; a Python package and a desktop GUI read them.

## Validation at a glance

Details and figures: [docs/VALIDATION.md](docs/VALIDATION.md).

- **Bowl crater (Ingersoll et al. 1992):** shadowed-floor temperature within 0.1 K (≤0.26%) of
  the analytic solution and shadow flux within 1.2%, at 15,840 faces.
- **Diviner equatorial diurnal cycle and Apollo 15/17 heat-flow probes:** every reference point
  that carries an uncertainty lies within its ±5 K bar (largest residual 4.91 K). The reference
  values come from the Hayne et al. (2017) compilation, which shares an author with PTM3D.
- **Diviner seasonal temperatures at a Haworth cold-trap site (240 m cells):** on the cold-trap
  floor (n = 2,970) the median summer-maximum bias is −0.80 K (IQR 11.19 K) with sub-facet
  roughness, and −10.30 K (IQR 10.38 K) without it.

## Citation

Please cite the software ([CITATION.cff](CITATION.cff)) and the supporting datasets on Zenodo:
concept DOI [10.5281/zenodo.21693403](https://doi.org/10.5281/zenodo.21693403) (CC-BY-4.0).
Heritage methods: Hayne et al. (2017, [10.1002/2017JE005387](https://doi.org/10.1002/2017JE005387));
Hayne & Aharonson (2015, [10.1002/2015JE004887](https://doi.org/10.1002/2015JE004887));
Paige et al. (2010, [10.1126/science.1187726](https://doi.org/10.1126/science.1187726)).
H-matrix view factors: Potter et al. (2023, [10.1016/j.jcpx.2023.100130](https://doi.org/10.1016/j.jcpx.2023.100130)).

## License and support

MIT ([LICENSE](LICENSE)). Questions and bug reports: the repository's Issues tab. For a build or
run problem, include your OS and compiler, the CMake configure summary, and the first 40 lines
of the run log. Contact: Paul Hayne (paul.hayne@colorado.edu).

Acknowledgments: Sam Potter (H-matrix algorithms, [fluxpy](https://github.com/sampotter/fluxpy));
NASA NAIF (SPICE); Intel (Embree); the LRO Diviner team; NASA Lunar-VISE, Europa Clipper, PSIE
and the Moon3D project.
