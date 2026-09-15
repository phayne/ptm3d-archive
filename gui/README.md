# PTM3D GUI

Modern PyQt6-based graphical user interface for the Planetary Thermophysical Model simulation suite.

## Supported entry point

**`python run_gui.py` is the ONE supported way to launch the PTM3D GUI** (the
thermal simulation application, `ptm3d_gui.py`). Run it from the `gui/`
directory:

```bash
cd gui/
python run_gui.py
```

An optional multi-module launcher (`run_dashboard.py`) bundles the DEM/mesh
tools alongside the thermal app; it is a convenience wrapper and gets less
testing than `run_gui.py`.

Before a simulation launches, the GUI now runs a pre-flight check
(`gui_core/preflight.py`): paths must exist and be readable, cloud-synced
paths (`/Library/CloudStorage/...`) require an explicit override, and
"dataless" cloud placeholder files (evicted from local disk) block the launch
— reading one hangs the engine indefinitely. Keep simulation input data on a
local disk, not in a cloud-synced folder.

## Features

### 1. Simulation Configuration

The GUI has **two levels**, and only the first is navigation. Clicking through
— body, run type, card, run — is how you get anywhere; the section list on
the left is **hidden until you ask for it**. Both levels edit the same
configuration.

- **Start page — select a body.** Five large buttons carrying the body's name
  and no other copy at all: Moon, Europa, Mercury, Asteroid, and Custom body.
  The three built-in bodies are exactly the ones the engine implements; the
  list is generated from `planetary_properties.h`, so a body it cannot run
  cannot have a button. What each one leads to is in its tooltip, and a body
  whose data support is thin says so on its own card — Mercury opens with a
  banner naming the datasets it does not ship. "Open a configuration file" and
  "Reproduce a published run" are the other two ways in.
- **Then — model domain and run type.** Global (an icosphere) or
  Local mesh (a terrain mesh you supply), and then Time series, Integrated
  statistics, or — for the Europa terrain grid — Parameter grid.
- **Then — one card**, sized to that body and that run type: eleven controls
  for a global Moon run, nine for a Europa terrain site, eight for Mercury.
  Its sections are Time window, the mesh, the surface, Radiation exchange,
  Thermal and Output — the solver's own settings, `thermal.use_gpu` among them,
  are in Thermal, and Output holds only where the files land. Each card states
  the exact keys it set, announces every automatic choice, and lists anything
  an opened file carries that the card does not show.
- **All settings (advanced)**: the complete reference view — Simulation, Mesh,
  Properties, Thermal, Output, Body, Illumination, Radiation, GPU and Engine
  Flags. It is **not** the primary navigation and it is not on screen by
  default. Open it from the button on any card, from the landing screen, or
  from *View → All Settings (Advanced)* (Ctrl+Shift+A); leave it with the
  **Back** button above the section list, or *View → Back to the Body Cards*
  (Ctrl+Shift+B). Leaving it returns you to the card you came from, not to the
  top of the flow, and the card re-reads the configuration on the way back so
  the two never disagree. Inside it, every section is always listed and
  "Advanced options" reveals the expert groups *inside* each section rather
  than hiding whole sections.

  Nothing about a saved configuration depends on whether this view is open:
  every section widget exists, loads and saves either way. Hiding it is a
  navigation change, not a behavior change.
- **YAML Preview**: the engine-facing configuration a run writes. Save does
  not write this: it edits your project file in place, changing only the keys
  you changed and preserving its comments (needs `ruamel.yaml`, which
  `requirements.txt` requires).
- **Validation**: the GUI's own checks, plus *Simulation → Check
  Configuration*, which hands the config to `ptm_thermal_sim
  --validate-config` and reports the engine's own words.

Why body-first: measured across the 78 configurations shipped in `configs/` and
`examples/`, **115 of the engine's 184 keys never take a second value anywhere**,
and the forty-five Europa terrain runs differ in exactly two scientific keys.
The tabbed view showed 145 controls for decisions that are between two and
seventeen keys wide. `scripts/gui_option_usage_audit.py` reproduces those
numbers.

Each card applies its defaults through the same mechanism the task cards used,
so "a card reports exactly the keys it wrote" is one rule with one set of
tests. Opening a file requires only that it be YAML and name a body under
`simulation.body`; everything else in it is loaded unchanged, listed above the
card, and editable under All settings. *File → Open* always lands on that
file's own body card, closing All settings if it was open.

The copy follows the same rule as the structure: a button says what it leads
to in one short line, and the detail is in its tooltip. The landing screen used
to carry a paragraph under each body button and two more under the baseline
entry; they are gone, and anything among them that was a genuine disclosure —
Mercury ships no albedo map, no topography and no worked example — now sits on
that body's own card, next to the controls it constrains.
`src/python/tests/test_gui_tooltip_coverage.py` reads every string the
application shows, on screen and in the source, and fails on internal
shorthand.

### 2. 3D Visualization (PyVista)
- Interactive 3D mesh rendering with zoom, pan, and rotate
- Multiple data fields: Temperature, Illumination, Solar flux
- Face picking with detailed information
- Timestep animation with playback controls
- Export high-resolution images
- **Coordinate readout** under the canvas, following the pointer

#### The coordinate readout

The line under the canvas names the facet under the pointer and its
planetocentric latitude and east-positive longitude. Those are the coordinates
the ENGINE wrote into `mesh.ply`, not a conversion the viewer performs, so they
are in whatever body-fixed frame the run simulated.

It shows nothing rather than a number when a number would be meaningless:

| What is under the pointer | What the readout says |
|---|---|
| A facet of a body-centered mesh whose radius matches a known body | `lat … ° S   lon … ° E · Moon, planetocentric, east-positive longitude · facet N` |
| A facet of a body-centered mesh of an unrecognized body (a shape model) | the same, labeled `body-fixed model frame` — latitude is real, and longitude is east-positive from whatever the shape model calls zero, which is not an IAU-ratified prime meridian |
| A local tangent-plane patch, built about its own center | `lat / lon: unavailable — this mesh has no georeferenced frame` |
| Empty space | `lat / lon: — (pointer is off the mesh)` |

A plausible wrong coordinate is worse than no coordinate, and nothing here
falls back to a number.

### 3. 2D Analysis (Matplotlib)
- Temperature distribution histograms
- Temperature vs. latitude scatter plots
- Time series analysis
- Min/max/mean statistics over time
- Individual face time series on selection

### 4. Results Browser
- Browse simulation output directories
- View available data fields and timesteps
- Load and compare multiple results
- Export to various formats (VTK, NumPy, HDF5, NetCDF, GeoTIFF, COG)

#### Export Data: the COG option

"Export Data" offers **COG** beside GeoTIFF, HDF5 and VTU/PVD. It writes Cloud
Optimized GeoTIFFs through the same `src/python/ptm_cog.py` that
`scripts/ptm3d_stats_to_cog.py` has produced delivered COGs with: single-band
Float32, GDAL COG layout, 512 px internal tiles, LZW, an AVERAGE overview
chain. One file per statistic under `cog/statistics/`, and — because a COG
holds one band — one file per timestep under `cog/<field>/` for a time series.

**Projection.** Auto-selected from the mesh: polar stereographic when the whole
mesh is beyond 70° latitude, equirectangular otherwise. The body radius comes
from the mesh's own median vertex radius. One choice is exposed, "Shared CRS:
central meridian 0°", and only on the polar branch, where it is the difference
between rasters that overlay across sites in a GIS session and rasters that do
not. The dialog states which meridian will be used either way.

**Resolution.** Auto-detected from the median face spacing, rounded to a fixed
table whose finest entry is 0.001° (about 30 m on the Moon). On a finer mesh
the dialog says so and names the mesh's own spacing; switch the unit selector
to m/px to export at that scale.

**Nodata, and the two sentinels.** A raster has one nodata value. PTM3D writes
two that do not mean the same thing: `NaN` ("this facet was not evaluated") and
`+inf` ("the Sun was still not visible at the top of the shadow-depth search
ceiling" — a measurement). Both become nodata in the main band, but the `+inf`
population is also written beside it as `<field>_unresolved.tif`, the fraction
of the facets behind each pixel that were unresolved, and is stamped out of the
main band so an unresolved area can never be read back as a finite depth. Every
band carries `PTM_FACES_TOTAL`, `PTM_FACES_FINITE`, `PTM_FACES_NAN`,
`PTM_FACES_POS_INF`, `PTM_FACES_NEG_INF` and `PTM_SENTINEL_POLICY`, and the
export README repeats the rule.

### 5. Console Output
- Real-time simulation progress
- Color-coded messages (info, success, warning, error)
- Auto-scroll with toggle
- Exportable log files

## Installation

### Prerequisites
- Python 3.9 or later
- PTM3D C++ simulation built and available

### Install Dependencies

```bash
cd gui/
pip install -r requirements.txt
```

### Required packages:
- `PyQt6` - GUI framework
- `pyvista` + `pyvistaqt` - 3D visualization
- `matplotlib` - 2D plotting
- `numpy` - Numerical operations
- `PyYAML` - Configuration file handling

### Optional packages (for enhanced features):
- `pyopencl` - GPU detection
- `h5py` - HDF5 export
- `netCDF4` - NetCDF export
- `rasterio` - GeoTIFF export of results
- `gdal` (`osgeo`) **and a system GDAL on PATH** (`gdaladdo`, `gdal_translate`)
  - COG export of results. Both are needed: the band is written through the
  Python bindings and the overview chain through the command-line tools,
  because only GDAL >= 3.12's CLI carries `OVERVIEW_RESAMPLING=AVERAGE` into
  the COG's `IMAGE_STRUCTURE` metadata. Without them the COG radio button is
  disabled and says why.
- `gdal` (`osgeo`) - GeoTIFF **DEM preview** in the dashboard's DEM module.
  Not pip-installable on its own; use `conda install -c conda-forge gdal`.
  The same GDAL install is the build-time prerequisite for the DTM tools the
  DEM module runs — see INSTALL.md, "DEM / DTM tools".

## Usage

### Launch the GUI

```bash
# From the gui/ directory
python run_gui.py

# Or with a project file
python run_gui.py --project /path/to/config.ptm3d

# Or with results to visualize
python run_gui.py --results /path/to/output/
```

### Quick Start

1. **Configure Simulation**
   - Select the "Configuration" tab
   - Set simulation parameters (body, duration, timesteps)
   - Configure mesh resolution
   - Set thermal and illumination options
   - Choose output directory

2. **Save and Run**
   - Save project (Ctrl+S)
   - Click "Run" button or press F5
   - Monitor progress in Console

3. **Visualize Results**
   - Results load automatically after simulation
   - Switch between 3D view and 2D plots
   - Use timestep slider or Play button for animation
   - Click faces in 3D view for details

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+N | New project |
| Ctrl+O | Open project |
| Ctrl+S | Save project |
| Ctrl+L | Load results |
| F5 | Run simulation |
| Shift+F5 | Stop simulation |
| F6 | Refresh visualization |
| Home | Reset camera |
| Ctrl+` | Toggle console |
| Ctrl+E | Export image |

## Project Structure

```
gui/
├── ptm3d_gui.py           # Main application (thermal simulation)
├── run_gui.py             # Launcher script — the supported entry point
├── dashboard.py           # Multi-module dashboard shell
├── run_dashboard.py       # Dashboard launcher (optional, less tested)
├── requirements.txt       # Python dependencies
├── README.md              # This file
│
├── gui_core/              # Core modules
│   ├── __init__.py
│   ├── config_manager.py  # Configuration handling
│   ├── styles.py          # Modern dark theme
│   └── simulation_runner.py # Background execution
│
├── gui_panels/            # UI panels
│   ├── __init__.py
│   ├── config_panel.py    # Configuration tabs
│   ├── visualization_panel.py  # 3D/2D views
│   ├── results_panel.py   # Results browser
│   └── console_panel.py   # Console output
│
├── dashboard_core/        # Dashboard shell: module registry + chaining
│   ├── module_interface.py  # Base class every dashboard module implements
│   ├── module_registry.py   # Discovery/registration of modules
│   ├── chaining_manager.py  # Hands one module's output to the next
│   └── sidebar.py           # Module navigation
│
├── module_adapters/       # Adapters wrapping each app as a dashboard module
│
├── dem_core/              # DEM module: logic (no Qt widgets)
│   ├── dem_config_manager.py # DEM tool settings
│   ├── dem_runner.py         # Runs dtm_to_ply_delaunay
│   └── dem_tooltips.py       # Flag-level help text
│
└── dem_panels/            # DEM module: UI panels (Preview / Convert / Inspect)
    ├── dem_preview_panel.py  # GeoTIFF preview + ROI picking (needs osgeo/GDAL)
    ├── dem_config_panel.py   # Nav rail + Conversion / ROI / QC options
    ├── dem_mesh_panel.py     # Mesh statistics and quality analysis
    └── dem_results_panel.py  # Generated-mesh summary
```

### DEM module prerequisites

`dem_runner.py` is a thin wrapper around one **compiled C++ binary built from
this repository**: `dtm_to_ply_delaunay` (DTM → PLY conversion, ROI extraction
and radial density grading). It looks in `build/bin/` first, then a legacy
`../dtm/` directory, then `PATH`. That target requires GDAL at build time;
without GDAL, CMake skips it and the Convert action reports a missing executable
with build instructions.

Two things are worth knowing before you use this module:

- **There is no mesh-refinement step, and no Refine page.** A separate
  `dtm_advanced_processor` binary was meant to provide adaptive refinement; it
  was never implemented (it wrote a zero-vertex mesh and exited 0) and has been
  removed, page and all. Use the **Convert** page with ROI enabled instead: it
  grades point density radially (Core Fraction / Density Exponent), which is
  what refinement was meant to achieve.
- **The Convert action is POSIX-only.** It streams tool output through
  `select.select()` on pipes, which does not work on Windows; the module says so
  rather than failing obscurely.

Full walkthrough, flag reference and known gaps:
[`docs/dem-to-mesh-workflow.md`](../docs/dem-to-mesh-workflow.md).

## Configuration File Format

Projects are saved as YAML files (`.ptm3d` or `.yaml`):

```yaml
simulation:
  start_time: "2030-04-10T00:00:00"
  duration_days: 3.55
  timesteps: 144
  body: Moon

mesh:
  resolution: 5
  import_file: ""
  scale_factor: 1.0

thermal:
  run_thermal_mesh: true
  threads: 0  # auto
  equilibration_orbits: 2
  use_gpu: false

radiation:
  enabled: false
  use_hmatrix: false

output:
  directory: output/my_simulation
  format: binary
  vtk_mode: both
```

## Troubleshooting

### PyVista not showing 3D view
- Ensure VTK is properly installed: `pip install vtk`
- Try: `pip install --upgrade pyvista pyvistaqt`
- On macOS, you may need: `pip install pyobjc`

### High DPI scaling issues
- Set environment variable: `QT_ENABLE_HIGHDPI_SCALING=1`
- Or: `QT_SCALE_FACTOR=1.5`

### Simulation executable not found
- Ensure PTM3D is built: `cd build && make`
- Check the build path in simulation_runner.py

### Memory issues with large meshes
- Reduce mesh resolution
- Disable statistics computation
- Use H-matrix compression for radiation

## Development

### Adding New Configuration Options

1. Add default value in `gui_core/config_manager.py`
2. Add widget in appropriate tab in `gui_panels/config_panel.py`
3. Map to C++ command-line flag in `gui_core/simulation_runner.py`

### Adding New Visualization Features

1. Extend `PyVistaWidget` in `gui_panels/visualization_panel.py`
2. Add controls to the visualization panel
3. Connect signals to main window

## Building a distribution package

Build a public/colleague tarball with `git archive`, which honors the
repository `.gitattributes` `export-ignore` rules:

```bash
git archive --format=tar.gz -o ptm3d.tar.gz HEAD
```

This excludes per-developer settings (`.claude/`, `gui/.claude/`). A raw
`tar`/`cp -r` of the working tree does **not** honor those rules and would leak
the excluded paths — use `git archive`. The Paul-only tools (batch runs,
exosphere, viewshed) are no longer staged in this repository at all: they were
deleted on 2026-08-03 (commit `2b5df5b`), so there is nothing left to exclude.

Some input datasets (SPICE kernels, DEMs, view factors) are large and are not
bundled; see the top-level data/distribution docs. If the GUI is run against a
cloud-synced folder whose files have been evicted to the cloud ("dataless"
placeholders), the pre-flight check will block the launch rather than let the
engine hang reading them — keep input data on a local disk.

## License

Part of the PTM3D simulation package.
