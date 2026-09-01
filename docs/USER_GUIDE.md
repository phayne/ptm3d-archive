# PTM3D User Guide

The working reference for running PTM3D: the simulation workflow, code
architecture, complete command-line reference, YAML configuration schema,
output formats, GUIs, and the Python post-processing tools. For installation
see [../INSTALL.md](../INSTALL.md); for a first run see
[../README.md](../README.md) and [../TUTORIAL.md](../TUTORIAL.md); for the
`PTM_*` environment variables see [ENVIRONMENT.md](ENVIRONMENT.md).

---

## Simulation Workflow

```
                              ┌─────────────────┐
                              │     INPUTS       │
                              └────────┬─────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
              v                        v                        v
     ┌────────────────┐      ┌────────────────┐      ┌────────────────┐
     │  Mesh File     │      │ SPICE Kernels  │      │  Config File   │
     │ (PLY/OBJ/VTK)  │      │  (Ephemeris)   │      │    (YAML)      │
     └───────┬────────┘      └───────┬────────┘      └───────┬────────┘
             │                       │                        │
             └───────────────────────┼────────────────────────┘
                                     │
                                     v
                        ┌────────────────────────┐
                        │    MESH GENERATION      │
                        │  Icosphere / DEM import │
                        │  Normals, areas, coords │
                        └───────────┬─────────────┘
                                    │
                                    v
                        ┌────────────────────────┐
                        │   ILLUMINATION MODEL    │
                        │  Ray tracing (Embree)   │
                        │  Direct solar flux      │
                        │  Primary reflection     │
                        │  Primary IR emission    │
                        │  Eclipse computation    │
                        └───────────┬─────────────┘
                                    │
                                    v
                        ┌────────────────────────┐
                        │     THERMAL MODEL       │
                        │  Time-stepped spin-up   │
                        │  Crank-Nicolson solver  │
                        │  T-dependent k(T), cp(T)│
                        └───────────┬─────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    v                               v
        ┌───────────────────┐           ┌───────────────────┐
        │ RADIATION EXCHANGE│           │   (no coupling)   │
        │  View factors     │           └─────────┬─────────┘
        │  H-matrix / CSR   │                     │
        │  Radiosity solver │                     │
        └─────────┬─────────┘                     │
                  │                               │
                  v                               │
        ┌───────────────────┐                     │
        │  COUPLING LOOP    │                     │
        │  T -> B -> Q_ind  │                     │
        │  Selective re-run │                     │
        │  Convergence check│                     │
        └────┬────┬─────────┘                     │
             │    │                               │
             │    └── iterate ──> back to         │
             │        Thermal Model               │
             │                                    │
             └────────────────┬───────────────────┘
                              │
                              v
                   ┌───────────────────┐
                   │  ICE STABILITY    │
                   │  Vapor diffusion  │
                   │  Newton-Raphson   │
                   │  Ice table depth  │
                   └────────┬──────────┘
                            │
                            v
                   ┌────────────────────┐
                   │      OUTPUTS       │
                   └────────┬───────────┘
                            │
           ┌────────────────┼────────────────┐
           │                │                │
           v                v                v
  ┌────────────────┐ ┌──────────────┐ ┌──────────────┐
  │  Binary Files  │ │   VTK/PVD    │ │ CSV Series   │
  │  temperature   │ │  ParaView    │ │ Locations    │
  │  illumination  │ │  compatible  │ │ Time series  │
  │  statistics    │ │  animation   │ │ Depth data   │
  └────────────────┘ └──────────────┘ └──────────────┘
```

---

## Architecture

### Source Layout

```
src/cpp/
├── main/             Main driver (ptm_thermal_sim.cpp)
├── thermal/          1D heat conduction, time-stepped equilibration
├── illumination/     Ray-traced solar flux with SPICE ephemeris
├── radiation/        View factors, H-matrix compression, radiosity
├── gpu/              OpenCL/HIP/CUDA acceleration
├── mesh/             Mesh I/O and geometry utilities
├── orbital/          Kepler propagation, body-fixed frames
├── maps/             Albedo/property FITS maps, spatial interpolation
├── constants/        Physical and planetary constants
├── ice/              Ice stability depth solver
├── tools/            Config parser, crater/DEM mesh generators
├── fileio/           Memory-mapped binary I/O
├── exosphere/        Sublimation, sputtering, ice evolution
└── viewshed/         Line-of-sight visibility analysis
```

### Core Modules

| Module | Key Classes | Description |
|--------|-------------|-------------|
| **thermal** | `thermal::Model`, `thermal::Profile`, `thermal::Config` | 1D Crank-Nicolson heat conduction with temperature-dependent properties, adaptive depth grids, and time-stepped equilibration with an adaptive substep |
| **illumination** | `IlluminationModel` | Shadow ray tracing (Embree), finite solar disk integration, primary-body reflected and thermal IR flux, eclipse geometry via SPICE |
| **radiation** | `RadiationExchangeManager`, `RadiositySolver`, `ViewFactorMatrix`, `HMatrixViewFactor` | Face-to-face view factors stored as a sparse matrix (compressed sparse row, "CSR", format) or in hierarchically compressed ("H-matrix") form; Jacobi/Gauss-Seidel radiosity iteration; adaptive thermal-radiation coupling |
| **gpu** | `ThermalOpenCLSolver`, `RadiosityOpenCLSolver`, `BVHRayTracer` | GPU-accelerated thermal solver (tridiagonal Thomas-algorithm solve), radiosity (sparse matrix–vector Jacobi iteration), and shadow ray tracing |
| **orbital** | `GenericBody`, `KeplerPropagator` | Keplerian orbit propagation, SPICE/Horizons ephemeris, body-fixed coordinate frames, spin state |
| **mesh** | `ptm::Mesh` | Icosphere generation, PLY/OBJ/VTK import, face normals/areas/centers, VTK export |
| **ice** | Ice stability solver | Vapor diffusion and Newton-Raphson ice table depth |

### Research Modules

| Module | Key Classes | Description |
|--------|-------------|-------------|
| **exosphere** | `ExospherePhysics`, `SputteringModel`, `IceLayerModel`, `IceCrystallizationModel` | Water vapor sublimation rates, ion sputtering yields, amorphous-to-crystalline ice phase transitions, regolith lag formation, composition-albedo feedback, multi-timescale epoch evolution |
| **viewshed** | Viewshed analyzer | Line-of-sight visibility from surface waypoints, terrain occlusion, mission traverse planning |

### Physics Highlights

**Thermal Model**:
- Temperature-dependent thermal conductivity: $k(T) = k_c \cdot (1 + \chi \cdot (T/350)^3)$
- Heat capacity: polynomial for lunar regolith; linear $c_p(T) = 7.49 \cdot T$ for water ice (Europa)
- Adaptive depth grid with logarithmic spacing
- Equilibration: the diurnal flux is integrated forward in time with the same Crank-Nicolson solver, over `thermal.equilibration_orbits` cycles, with an adaptive substep chosen from the per-layer Fourier mesh number. There is no spectral or harmonic shortcut, and no convergence test — the spin-up length is a user choice, and every shipped configuration is short of the deep column's relaxation time. See [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) section 1
- Validated against Lunar Diviner data (RMS error <4 K)

**Radiation Exchange**:
- Band-separated radiosity, on by default: a visible band $(I - AF)B_{\text{vis}} = AQ_{\text{solar}}$ scattering with the Bond albedo, and a thermal-IR band $(I - \rho F)B_{\text{ir}} = \varepsilon\sigma T^4$ scattering with $\rho = 1-\varepsilon$; the two absorbed fluxes are summed. `radiation.band_separated: false` recovers the single gray band
- View factor methods: centroid (far field) and subdivision (near field), selected adaptively by dimensionless separation $N_s = r/\sqrt{A}$
- Hierarchical-matrix ("H-matrix") compression with octree spatial partitioning, low-rank block compression by adaptive cross approximation (ACA), and $O(N \log N)$ matrix–vector multiplication cost
- Adaptive thermal-radiation coupling: selective face re-run, warm-started solver, convergence on the maximum per-face temperature change

**Exosphere Model**:
- Sublimation-driven water vapor loss with Hertz-Knudsen kinetics
- Ion sputtering yields (magnetospheric bombardment)
- Amorphous-to-crystalline ice transition (Arrhenius kinetics)
- Composition-albedo feedback: non-ice lag concentration lowers albedo, increasing temperature (positive feedback for thermal segregation)
- Multi-timescale operator splitting: diurnal, seasonal, and geological epochs

![Thermal Wave Profiles](figures/thermal_wave_profiles.png)
*Subsurface temperature profiles at different local solar times showing the diurnal thermal wave. Temperature evolution with depth and time; the white dashed line indicates the thermal skin depth (~3.4 cm).*

---

## Command-Line Reference

### Simulation Setup

```
--config <file>               Load YAML configuration file
--body, -b <name>             Body name. Moon, Mercury and Europa have built-in
                              properties; any other name needs a generic_body: block
--start <time>                Start time (ISO 8601, default: 2025-01-01T00:00:00)
--resolution <n>              Global icosphere resolution (default: 5)
--duration <days>             Simulation duration in days
--timesteps <n>               Number of output timesteps
--output <dir>                Output directory
--verbose                     Enable detailed logging
```

### Mesh

```
--mesh-file <file>            Import mesh (PLY, OBJ, VTK)
--mesh-scale <factor>         Scale imported mesh (default: 1.0)
--export-basic-mesh           Export basic mesh to VTK
```

### Thermal

```
--thermal-mesh                Run thermal model for all faces
--thermal-threads <n>         Number of parallel threads (0 = auto)
--equilibration-orbits <n>    Diurnal cycles for equilibration (default: 1)
--use-gpu-thermal             Use GPU thermal solver (OpenCL)
--geothermal-heat-flow <Q>    Bottom boundary heat flow (W/m^2)
--thermal-m <n>               Layers in upper skin depth (default: 10)
--thermal-n <n>               Layer growth factor (default: 15)
--thermal-b <n>               Skin depths to bottom (default: 15)
```

### Physical properties

Two scalar properties can be set from the command line on any body, built-in or
generic, without editing the YAML. Both go through the same override channel the
solvers already honor (`thermophysical.albedo_override`, `ks_override`,
`kd_override`, `H_override`), so the run prints `[Thermo override] applied at solver: ...`
confirming the value reached the solver and not merely `main()`.

```
--albedo <A0>                 Bond albedo A0, in [0, 1)
--thermal-inertia <TI>        Nominal surface thermal inertia [J m^-2 K^-1 s^-1/2]
--constant-thermal-inertia <TI>   Thermal inertia held exactly constant
```

`--thermal-inertia TI` means

```
TI = sqrt(ks * rhos * cp(273 K))        ->   ks = TI^2 / (rhos * cp(273 K))
```

with the densities and the heat capacity left at the body's values, the
conductivity carrying the whole change, and `kd` scaled by the same factor so
the `kd/ks` depth contrast survives. This is the convention already used by
`scripts/ema_campaign_driver.py`, `scripts/divval_sweep.py` and the shipped pair
`examples/quickstart_asteroid_diurnal.yaml` /
`examples/quickstart_asteroid_diurnal_ti200_check.yaml` — and running the first
of those with `--thermal-inertia 200` reproduces the second to 1e-4 K.

`cp` is evaluated at 273 K by the body's own `cp(T)` polynomial, the same
reference the engine's TI(273 K) property maps use (Hayne et al. 2017). On a
body with a constant `cp` — every generic body — that is `cp0` exactly. On the
Moon it is the only workable choice: `cp0` there is the *constant term* of the
polynomial and is negative (−3.6125 J/kg/K).

The flag does **not** make thermal inertia constant. `cp(T)` and `chi` are left
alone, so on a body that has them the delivered TI moves with temperature about
the requested value and the run says so. `--constant-thermal-inertia` is the
other reading: it derives `cp = TI^2/(ks*rhos)` and forces `chi = 0`, `kd = ks`,
`rhod = rhos`, making TI exactly constant in temperature and depth at the cost
of the two-layer profile. The two are mutually exclusive and the run refuses if
both are given.

`--albedo` is out-ranked by a per-face albedo map for the absorbed-solar flux;
the run warns when one is configured. Set `illumination.use_albedo_map: false`
to sweep albedo directly. With `incidence_angle_albedo: true` the Hapke angular
model is re-anchored on the new A0 (the single-scattering albedo `w` is
re-derived), so the flag stays consistent with that gate rather than bypassing
it.

### Radiation Exchange

```
--radiation-exchange          Enable inter-face radiation exchange
--rad-hmatrix                 Use H-matrix compression (recommended for >50K faces)
--rad-use-gpu                 Use GPU radiosity solver (OpenCL)
--rad-view-factor-file <f>    Save/load pre-computed view factors
--rad-coupling-iter <n>       Max coupling iterations (default: 10)
--rad-coupling-temp-tol <K>   Temperature convergence threshold (default: 0.5 K)
--rad-coupling-flux-tol <W>   Flux threshold, legacy selection only (default: 0.1 W/m^2)
--rad-coupling-face-frac <f>  Face fraction still changing (default: 0.01)
--rad-aca-tolerance <t>       ACA compression tolerance (default: 1e-3)
--rad-hmatrix-threads <n>     H-matrix build threads (0 = auto)
```

### Illumination

```
--albedo-map <file>           Custom albedo FITS map
--no-primary-reflection       Disable reflected light from primary body
--shadow-flux                 Enable shadow/crater scattered flux
--positions <file>            Export body positions to CSV
```

### Output

```
--output-format <fmt>         Output format: "binary" or "vtk" (default: binary)
--vtk-mode <mode>             VTK mode: "thermal", "complete", "both"
--include-primary-ir          Include primary IR in VTK output
--locations <file>            Extract illumination for lat/lon locations
--depths <file>               Export subsurface temperature profiles
--series-output <file>        Time series output file
```

Run `./build/bin/ptm_thermal_sim --help` for the complete reference.

---

## Configuration File

PTM3D accepts YAML configuration files via `--config`. Command-line flags override config file values. The top-level sections are:

```yaml
simulation:
  body: "Europa"
  start_time: "2025-01-01T00:00:00"
  duration_days: 3.55
  timesteps: 144

mesh:
  resolution: 5
  import_file: ""           # PLY/OBJ/VTK path (overrides icosphere)
  use_topographic: false    # Use topographic DEM

thermal:
  run_thermal_mesh: true
  threads: 0                # 0 = auto-detect
  equilibration_orbits: 4   # default: 1
  geothermal_heat_flow: -1  # -1 = planet default

illumination:
  include_primary_reflection: true
  use_batch_processing: true
  albedo_maps_directory: "data/albedo_maps"

radiation:
  enabled: false
  use_hmatrix: false
  max_coupling_iterations: 16
  coupling_temp_threshold: 0.5   # K
  coupling_flux_threshold: 0.1   # W/m^2
  coupling_mode: bounded         # bounded | fixed | legacy (see
                                 # docs/radiation-exchange.md 7.2.1).
                                 # `legacy` reproduces pre-2026-08-05 runs and
                                 # makes the answer depend on the chunk plan.
  coupling_select_dt: 0.0        # K; 0 = re-thermal every face carrying
                                 # indirect flux (decomposition-independent)
  exchange_selection: all        # all | shadow (see
                                 # docs/radiation-exchange.md 7.2.2).
                                 # `shadow` reproduces pre-2026-08-05 runs: it
                                 # drops faces that are lit for most of THIS
                                 # window out of radiation exchange entirely,
                                 # so a chunked run zeroes their indirect flux.
  min_shadow_fraction: 0.3       # exchange_selection: shadow only

output:
  directory: "output/simulation"
  format: "binary"

generic_body:                # For arbitrary bodies
  name: "MyAsteroid"
  radius_km: 50.0
  semi_major_axis_au: 2.5
  spin_period_hours: 6.0
  albedo: 0.1
  emissivity: 0.95
  # ... (see configuration.md for full reference)
```

See [`../examples/`](../examples/) for runnable configurations (relative paths
only; start with `quickstart_moon_diurnal.yaml`). The study configurations under
`../configs/` are kept for provenance and may reference datasets and paths that
are not distributed.

---

## Output Formats

PTM3D generates several output types:

```
output/
├── mesh.ply                    # Mesh geometry
├── statistics/
│   ├── temperature_avg.bin     # Mean temperature per face
│   ├── temperature_max.bin     # Maximum temperature
│   ├── temperature_min.bin     # Minimum temperature
│   ├── illumination_avg.bin    # Mean solar flux
│   └── ice_depth.bin           # Ice stability depth
├── temperature/
│   ├── temperature_00001.bin   # Per-timestep temperatures
│   └── ...
└── illumination_series.csv     # Location time series (if --locations)
```

**Binary format**: each statistics file is a 4-byte little-endian int32 face
count followed by one single-precision float per mesh face (see the tutorial
`tutorials/06_interpreting_outputs.md` for a reader snippet).

**Time index**: per-step frames are labeled by output index, and index `NNNN`
means the same thing in every one of them. `temperature/temperature_NNNN.bin`
holds the surface state **at** the epoch on line `NNNN` of `time_steps.txt`,
which is the same epoch at which `direct_solar/direct_solar_NNNN.bin` and
`illumination/illumination_NNNN.bin` are sampled. So index 0 is the state the
run started from — the spin-up seed, or the checkpoint of a warm-started chunk
— and no frame reflects flux that arrives after its own epoch. Joining
temperature to illumination by index is therefore safe, and a chunked run
merged with `ptm_merge_chunks` tiles the epochs exactly once with no gap and no
duplicate. To confirm this on a build, run
`scripts/check_output_time_alignment.py` (it needs an anchor outside the
series — a warm-start checkpoint or a `PTM_DUMP_SEED` file — because a uniform
shift of the whole series is invisible from inside it).

**VTK/PVD export**: Use `python src/python/vtk_converter.py` to create ParaView-compatible time series.

**Additional exports**: NetCDF (`netcdf_converter.py`), HDF5 (`ptm_hdf5.py`), GeoTIFF (`ptm_geotiff.py`).

---

## Graphical User Interface

PTM3D ships one supported PyQt6 GUI, plus an optional dashboard shell that
bundles additional modules:

| Entry point | Script | Purpose |
|-----|--------|---------|
| **PTM3D Main** (supported) | `gui/run_gui.py` | Thermal simulation configuration, 3D visualization, results browsing |
| **Dashboard** (optional) | `gui/run_dashboard.py` | Multi-module shell; adds the **DEM module** (GeoTIFF DTM → PLY mesh, ROI extraction) alongside the thermal app. Convenience wrapper, less tested than `run_gui.py`. |

The standalone `exosphere_gui.py` and `viewshed_gui.py` scripts listed in
earlier revisions of this guide **no longer exist**. Run the exosphere model
through `build/bin/ptm_exosphere_main`; the viewshed capability is the
`src/cpp/viewshed/` module, whose command-line front end is
`build/bin/dark_horse_viewshed`.

The DEM module needs two compiled binaries (`dtm_to_ply_delaunay`,
`dtm_advanced_processor`) that require GDAL to build — see
[dem-to-mesh-workflow.md](dem-to-mesh-workflow.md) for the full walkthrough and
[../INSTALL.md](../INSTALL.md) ("DEM / DTM tools") for the prerequisites.

### Quick Start

```bash
cd gui/
pip install -r requirements.txt
python run_gui.py
```

### Features

- **Tabbed configuration**: Simulation, mesh, thermal, radiation, illumination, and output settings
- **3D visualization**: Interactive PyVista-based mesh rendering with temperature colormap
- **2D analysis**: Histograms, latitude plots, and time series
- **Live preview**: Real-time YAML configuration preview
- **Console output**: Color-coded simulation progress and logging

See [../gui/README.md](../gui/README.md) for detailed GUI documentation.

---

## Python Tools

### The `ptm` package and the `ptm` command

`pip install -e .` from the repository root installs one importable package and
one console script. Everything below is plumbing and presentation — the C++
engine is still the model.

```bash
ptm info  output/quickstart_moon_diurnal            # what the run records
ptm plot  output/quickstart_moon_diurnal map --save map.png
ptm run   examples/quickstart_moon_diurnal.yaml     # one simulation
ptm sweep examples/quickstart_asteroid_diurnal.yaml \
          --vary thermal_inertia=50,100,200,400     # a grid of them
```

```python
from ptm import Results, Sweep

run = Results("output/quickstart_moon_diurnal")
run.plot_globes("temperature", "max", save="globes.png")

sweep = Sweep("examples/quickstart_asteroid_diurnal.yaml",
              grid={"thermal_inertia": [50, 100, 200, 400]})
sweep.run()                                          # serial; resumes
ti   = sweep.values("thermal_inertia")               # the axis
tmax = sweep.collect(lambda r: r.stat("temperature", "max").max())
sweep.plot_curves(save="family.png")
```

#### Sweeps

| Behavior | Detail |
|---|---|
| Execution | Serial by default. `jobs=N` runs `N` engine processes at once and **refuses to start** if any point asks for the GPU — there is one GPU, and concurrent use of it wedges the device. |
| Resume | Each point writes `sweep_point.json` before it starts and rewrites it when it finishes. A re-run skips a point only if that file says `complete`, the config still hashes the same, **and** the outputs are on disk; every skip and every re-run prints its reason. |
| Provenance | The swept values, the generated overrides, the argv, the engine path and md5, and the config hash are recorded next to the outputs. `ptm.sweep.load_sweep(dir)` reopens a finished sweep without retyping the grid. |
| Generated configs | Written to each point's own output directory, never into `examples/` or `configs/`. `radiation.view_factor_file` is made absolute, because it is the one key the engine resolves against the YAML's own directory (`config_parser.cpp:507-517`). |
| Axis names | `albedo` and `constant_thermal_inertia` go through the engine CLI flags `--albedo` / `--constant-thermal-inertia` when the binary advertises them (detected from `--help`, never assumed) and otherwise through the config keys `thermophysical.albedo_override` / `thermophysical.constant_thermal_inertia`, which the engine interprets identically (measured: byte-identical output either way). `thermal_inertia` is **flag-only** — see below. Any name containing a dot is a config key. A bare name that is neither must be declared in `label_axes=`, so it cannot become a dead key in the YAML. |
| No engine needed to read one | Constructing a sweep, `--list`, `--dry-run`, `table()`, `write_table()`, `grid_of()`, the plots and `load_sweep()` all work with no compiled binary present. The engine is resolved only when something launches. |

#### Thermal inertia is two parameters — say which one

TI is derived here (the primaries are `ks`/`kd`/`rhos`/`rhod`/`cp0`/`chi`), so
"set TI to 200" has two answers and the engine implements both. They are
**different physics**, not two spellings.

| Grid key | What the engine does | Route |
|---|---|---|
| `thermal_inertia` | `ks = TI²/(rhos·cp(273 K))`, `kd` scaled by the same factor. Densities, `cp(T)` and `chi` untouched, so the depth contrast is preserved. Resolves onto `thermophysical.{ks,kd}_override`. This is the convention already used by `ema_campaign_driver.py`, `divval_sweep.py` and the shipped `quickstart_asteroid_diurnal_ti200_check.yaml`. | `--thermal-inertia` **only** |
| `constant_thermal_inertia` | `cp = TI²/(ks·rhos)` instead, plus `chi=0`, `kd=ks`, `rhod=rhos`, so TI is exactly constant in temperature **and** depth — and the thermal diffusivity moves along the axis where the other holds it. | `--constant-thermal-inertia`, or the config key of the same name |

The two are far enough apart to change a thermal inertia you fit, so the choice
is not a formality. How far apart is a property of the body, the mesh and the
window, not a constant worth carrying between runs: measure it on the case you
care about by running the same config twice, once per key, and comparing a
statistic you state in full — the metric, the faces it covers and the window it
covers. "Diurnal amplitude" on its own does not pin those down, and different
choices of them disagree. A fully specified example: the median, over all
faces, of `temperature_max - temperature_min` from `statistics/`, accumulated
over the run's final diurnal cycle. The engine refuses a run given both, and so
does `Sweep`.

`thermal_inertia` has no config-file route on purpose. Building one would mean
this Python layer evaluating the body's `cp` polynomial at 273 K, which is
physics the wrapper does not do — and on the Moon `cp0` is that polynomial's
*constant term*, −3.6125 J/kg/K, so the naive `sqrt(ks·rhos·cp0)` is not a real
number. On an engine without the flag you get an error naming
`constant_thermal_inertia`, never a silent switch to different physics.
| Collection | `values()`, `collect()`, `stat()`, `grid_of()`, `table()`, `write_table()`, `curves()`. `table()` returns a list of dicts — `pandas.DataFrame(sweep.table(...))` works, and nothing here requires pandas. |

Worked examples: [`examples/python/fit_thermal_inertia.py`](../examples/python/fit_thermal_inertia.py)
(sweep → lookup table → fit thermal inertia to a 233 GHz brightness-temperature
curve, with the observable conversion supplied as a callable) and
[`examples/python/sweep_replaces_45_configs.py`](../examples/python/sweep_replaces_45_configs.py)
(the 40-file `configs/europa_lat*` product as one sweep, checked key-by-key
against the shipped files).

### Post-Processing Scripts (`src/python/`)

| Script | Purpose |
|--------|---------|
| `vtk_converter.py` | Convert binary output to VTK/PVD for ParaView |
| `vtk_converter_exosphere.py` | Convert exosphere output to VTK |
| `netcdf_converter.py` | Export to NetCDF |
| `ptm_hdf5.py` | Export to HDF5 |
| `ptm_geotiff.py` | Export to GeoTIFF (geospatial) |
| `plot_depth_profiles.py` | Subsurface temperature-depth profiles |
| `extract_timeseries.py` | Extract data for specific lat/lon locations |
| `lunar_validation_figure.py` | Validation plots vs. Hayne et al. (2017) |
| `europa_visualization.py` | Europa-specific visualization |
| `exosphere_analysis.py` | Exosphere output analysis |
| `thermal_vis.py` | Thermal field visualization |
| `viewshed_pv_animator.py` | Animated viewshed in ParaView |
| `ptm_paraview_converter.py` | ParaView format conversion |

### Analysis Scripts (`scripts/`)

Research analysis scripts for ice shell modeling, polar wander, millimeter-wavelength emission modeling, and seasonal thermal surveys.

### Python Packages (`src/python/`)

| Package | Purpose |
|---------|---------|
| `ojakangas_stevenson/` | Equilibrium ice shell thickness model (O&S 1989) |
| `ptm_ice/` | Ice shell thermal analysis |
| `horizons/` | JPL Horizons ephemeris API wrapper |
| `damit/` | DAMIT asteroid shape model database |

### Example: Convert and Visualize

```bash
# Convert to ParaView format
python src/python/vtk_converter.py output/simulation output/paraview

# Export to NetCDF
python src/python/netcdf_converter.py \
    --input output/paraview/statistics.pvd \
    --output output/statistics.nc \
    --format pvd
```

### Extract Time Series for Specific Locations

```bash
# Create locations file (latitude longitude name)
cat > locations.txt << EOF
0.0 0.0 SubsolarPoint
45.0 0.0 MidLatitude
85.0 0.0 NearPole
EOF

./build/bin/ptm_thermal_sim \
    --config examples/quickstart_moon_diurnal.yaml \
    --locations locations.txt \
    --series-output illumination_series.csv
```
